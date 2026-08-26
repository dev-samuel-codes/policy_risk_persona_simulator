"""대용량 Parquet 페르소나를 위한 스트리밍 조회 도구.

후보 목록을 만들거나 소수 UUID를 해석하기 위해 데이터셋 전체를 pandas
DataFrame으로 올리지 않는다. 지역 옵션은 두 컬럼만 순회해 프로세스 안에서
캐시하고, 후보/UUID 조회는 필요한 컬럼과 row group만 읽는다.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backend.ai_simulation_core.personas.persona_downloader import (
    get_local_parquet_files,
)
from backend.ai_simulation_core.personas.persona_sampler import PERSONA_COLUMNS

RegionScope = Literal["nationwide", "specific"]
AgeCohort = Literal["eligible", "boundary"]

REQUIRED_DISPLAY_COLUMNS = [
    "uuid",
    "occupation",
    "sex",
    "age",
    "province",
    "district",
]
DISPLAY_COLUMNS = [
    *REQUIRED_DISPLAY_COLUMNS,
    "persona",
    "professional_persona",
    "family_persona",
]
REGION_COLUMNS = ["province", "district"]
SCAN_BATCH_SIZE = 4096
PUBLIC_SERVANT_KEYWORD = "공무원"


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "item"):
        value = value.item()
    return value


def _clean_text(value: Any) -> str:
    return str(_clean_scalar(value) or "").strip()


def _clean_age(value: Any) -> int | None:
    value = _clean_scalar(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _available_columns(parquet_file: Path, requested: Iterable[str]) -> list[str]:
    available = set(pq.ParquetFile(parquet_file).schema.names)
    return [column for column in requested if column in available]


def _dataset_signature(
    parquet_files: Sequence[Path],
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in parquet_files
    )


@lru_cache(maxsize=4)
def _cached_region_options(
    signature: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    districts_by_province: dict[str, set[str]] = {}

    for filename, _, _ in signature:
        parquet_file = Path(filename)
        columns = _available_columns(parquet_file, REGION_COLUMNS)
        if "province" not in columns or "district" not in columns:
            continue

        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(
            columns=columns,
            batch_size=SCAN_BATCH_SIZE,
            use_threads=False,
        ):
            provinces = batch.column(batch.schema.get_field_index("province"))
            districts = batch.column(batch.schema.get_field_index("district"))
            for province_value, district_value in zip(
                provinces.to_pylist(), districts.to_pylist(), strict=True
            ):
                province = _clean_text(province_value)
                district = _clean_text(district_value)
                if not province:
                    continue
                districts_by_province.setdefault(province, set())
                if district:
                    # district는 "서울-서초구" 같은 원본 값을 그대로 보존한다.
                    districts_by_province[province].add(district)

    return tuple(
        (province, tuple(sorted(districts)))
        for province, districts in sorted(districts_by_province.items())
    )


def clear_persona_catalog_caches() -> None:
    """테스트 또는 데이터셋 교체 뒤 지역 옵션 캐시를 비운다."""

    _cached_region_options.cache_clear()


def get_region_options(*, auto_download: bool = True) -> dict[str, list[dict]]:
    parquet_files = get_local_parquet_files(auto_download=auto_download)
    if not parquet_files:
        raise FileNotFoundError("페르소나 Parquet 파일을 찾을 수 없습니다.")

    signature = _dataset_signature(parquet_files)
    options = _cached_region_options(signature)
    return {
        "provinces": [
            {"province": province, "districts": list(districts)}
            for province, districts in options
        ]
    }


def _validate_filter(
    *,
    region_scope: RegionScope,
    province: str,
    district: str,
    age_min: int | None,
    age_max: int | None,
) -> None:
    if region_scope not in {"nationwide", "specific"}:
        raise ValueError("region_scope는 nationwide 또는 specific이어야 합니다.")
    if region_scope == "specific" and not province:
        raise ValueError("특정 지역 조회에는 province가 필요합니다.")
    if age_min is not None and age_max is not None and age_min > age_max:
        raise ValueError("age_min은 age_max보다 클 수 없습니다.")
    if region_scope == "nationwide" and (province or district):
        raise ValueError(
            "province와 district는 특정 지역 조회에서만 사용할 수 있습니다."
        )


def region_matches(
    persona: dict,
    *,
    region_scope: RegionScope,
    province: str = "",
    district: str = "",
) -> bool:
    if region_scope == "nationwide":
        return True
    if _clean_text(persona.get("province")) != province:
        return False
    return not district or _clean_text(persona.get("district")) == district


def classify_age(
    age: Any,
    *,
    age_min: int | None,
    age_max: int | None,
) -> tuple[AgeCohort | None, str]:
    normalized_age = _clean_age(age)
    if normalized_age is None:
        return None, "age_unavailable"

    if age_min is None and age_max is None:
        return "eligible", "no_age_restriction"

    lower_match = age_min is None or normalized_age >= age_min
    upper_match = age_max is None or normalized_age <= age_max
    if lower_match and upper_match:
        return "eligible", "within_range"
    if age_min is not None and normalized_age == age_min - 1:
        return "boundary", "below_minimum"
    if age_max is not None and normalized_age == age_max + 1:
        return "boundary", "above_maximum"
    return None, "outside_allowed_cohorts"


def persona_selection_match(
    persona: dict,
    *,
    region_scope: RegionScope,
    province: str = "",
    district: str = "",
    age_min: int | None = None,
    age_max: int | None = None,
) -> dict[str, Any]:
    matches_region = region_matches(
        persona,
        region_scope=region_scope,
        province=province,
        district=district,
    )
    cohort, reason = classify_age(persona.get("age"), age_min=age_min, age_max=age_max)
    return {
        "region_match": matches_region,
        "age_cohort": cohort,
        "age_match_reason": reason,
    }


def _row_to_dict(batch: pa.RecordBatch, row_index: int) -> dict[str, Any]:
    return {
        name: _clean_scalar(batch.column(column_index)[row_index])
        for column_index, name in enumerate(batch.schema.names)
    }


def _candidate_matches(
    persona: dict,
    *,
    region_scope: RegionScope,
    province: str,
    district: str,
    age_min: int | None,
    age_max: int | None,
    cohort: AgeCohort,
) -> dict[str, Any] | None:
    occupation = _clean_text(persona.get("occupation"))
    if not occupation or PUBLIC_SERVANT_KEYWORD in occupation:
        return None

    match = persona_selection_match(
        persona,
        region_scope=region_scope,
        province=province,
        district=district,
        age_min=age_min,
        age_max=age_max,
    )
    if not match["region_match"] or match["age_cohort"] != cohort:
        return None
    return match


def get_persona_candidates(
    *,
    region_scope: RegionScope = "nationwide",
    province: str = "",
    district: str = "",
    age_min: int | None = None,
    age_max: int | None = None,
    cohort: AgeCohort = "eligible",
    limit: int = 12,
    seed: int = 0,
    auto_download: bool = True,
) -> list[dict[str, Any]]:
    province = province.strip()
    district = district.strip()
    _validate_filter(
        region_scope=region_scope,
        province=province,
        district=district,
        age_min=age_min,
        age_max=age_max,
    )
    if cohort not in {"eligible", "boundary"}:
        raise ValueError("cohort는 eligible 또는 boundary이어야 합니다.")
    if not 1 <= limit <= 24:
        raise ValueError("limit은 1 이상 24 이하여야 합니다.")
    if cohort == "boundary" and age_min is None and age_max is None:
        return []

    parquet_files = get_local_parquet_files(auto_download=auto_download)
    if not parquet_files:
        raise FileNotFoundError("페르소나 Parquet 파일을 찾을 수 없습니다.")

    work_units: list[tuple[Path, int]] = []
    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        work_units.extend(
            (parquet_file, row_group_index)
            for row_group_index in range(parquet.num_row_groups)
        )

    rng = random.Random(seed)
    rng.shuffle(work_units)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for parquet_file, row_group_index in work_units:
        columns = _available_columns(parquet_file, DISPLAY_COLUMNS)
        if not set(REQUIRED_DISPLAY_COLUMNS).issubset(columns):
            continue
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(
            row_groups=[row_group_index],
            columns=columns,
            batch_size=SCAN_BATCH_SIZE,
            use_threads=False,
        ):
            matching_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row_index in range(batch.num_rows):
                persona = _row_to_dict(batch, row_index)
                persona_id = _clean_text(persona.get("uuid"))
                if not persona_id or persona_id in seen_ids:
                    continue
                match = _candidate_matches(
                    persona,
                    region_scope=region_scope,
                    province=province,
                    district=district,
                    age_min=age_min,
                    age_max=age_max,
                    cohort=cohort,
                )
                if match is not None:
                    matching_rows.append((persona, match))

            # 같은 seed에서는 재현되고 seed가 바뀌면 후보 순서도 달라진다.
            rng.shuffle(matching_rows)
            for persona, match in matching_rows:
                persona_id = _clean_text(persona.get("uuid"))
                seen_ids.add(persona_id)
                candidates.append({**persona, "match": match})
                if len(candidates) >= limit:
                    return candidates

    # 조건을 충족하지 못할 때 다른 지역/나이로 대체하지 않는다.
    return candidates


def _find_uuid_locations(
    parquet_files: Sequence[Path], wanted_ids: set[str]
) -> tuple[dict[str, tuple[Path, int, int]], set[str]]:
    locations: dict[str, tuple[Path, int, int]] = {}
    remaining = set(wanted_ids)

    for parquet_file in parquet_files:
        if not remaining:
            break
        parquet = pq.ParquetFile(parquet_file)
        if "uuid" not in parquet.schema.names:
            continue
        for row_group_index in range(parquet.num_row_groups):
            if not remaining:
                break
            row_offset = 0
            for batch in parquet.iter_batches(
                row_groups=[row_group_index],
                columns=["uuid"],
                batch_size=SCAN_BATCH_SIZE,
                use_threads=False,
            ):
                uuid_array = batch.column(0)
                # is_in은 UUID 한 컬럼에만 적용해 Python 객체 생성을 줄인다.
                mask = pc.is_in(uuid_array, value_set=pa.array(sorted(remaining)))
                for local_index in pc.indices_nonzero(mask).to_pylist():
                    persona_id = _clean_text(uuid_array[local_index])
                    if persona_id in remaining:
                        locations[persona_id] = (
                            parquet_file,
                            row_group_index,
                            row_offset + int(local_index),
                        )
                        remaining.remove(persona_id)
                row_offset += batch.num_rows
                if not remaining:
                    break
    return locations, remaining


def _read_personas_at_locations(
    locations: dict[str, tuple[Path, int, int]],
) -> dict[str, dict[str, Any]]:
    by_unit: dict[tuple[Path, int], dict[int, str]] = {}
    for persona_id, (parquet_file, row_group_index, row_offset) in locations.items():
        by_unit.setdefault((parquet_file, row_group_index), {})[row_offset] = persona_id

    resolved: dict[str, dict[str, Any]] = {}
    for (parquet_file, row_group_index), offsets in by_unit.items():
        columns = _available_columns(parquet_file, PERSONA_COLUMNS)
        parquet = pq.ParquetFile(parquet_file)
        batch_start = 0
        max_offset = max(offsets)
        for batch in parquet.iter_batches(
            row_groups=[row_group_index],
            columns=columns,
            batch_size=SCAN_BATCH_SIZE,
            use_threads=False,
        ):
            batch_end = batch_start + batch.num_rows
            for row_offset, persona_id in offsets.items():
                if batch_start <= row_offset < batch_end:
                    resolved[persona_id] = _row_to_dict(batch, row_offset - batch_start)
            if batch_end > max_offset:
                break
            batch_start = batch_end
    return resolved


def resolve_personas(
    persona_ids: Sequence[str], *, auto_download: bool = True
) -> list[dict[str, Any]]:
    normalized_ids = [_clean_text(persona_id) for persona_id in persona_ids]
    if any(not persona_id for persona_id in normalized_ids):
        raise ValueError("페르소나 ID는 비어 있을 수 없습니다.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("페르소나 ID는 중복될 수 없습니다.")

    parquet_files = get_local_parquet_files(auto_download=auto_download)
    if not parquet_files:
        raise FileNotFoundError("페르소나 Parquet 파일을 찾을 수 없습니다.")

    locations, missing = _find_uuid_locations(parquet_files, set(normalized_ids))
    if missing:
        raise ValueError(
            "존재하지 않는 페르소나 ID가 있습니다: " + ", ".join(sorted(missing))
        )
    resolved = _read_personas_at_locations(locations)
    # Parquet 파일/row group 순서가 아니라 요청 순서를 보존한다.
    return [resolved[persona_id] for persona_id in normalized_ids]


def validate_persona_selection(
    personas: Sequence[dict[str, Any]],
    *,
    region_scope: RegionScope,
    province: str = "",
    district: str = "",
    age_min: int | None = None,
    age_max: int | None = None,
) -> list[dict[str, Any]]:
    province = province.strip()
    district = district.strip()
    _validate_filter(
        region_scope=region_scope,
        province=province,
        district=district,
        age_min=age_min,
        age_max=age_max,
    )

    evidence: list[dict[str, Any]] = []
    for persona in personas:
        persona_id = _clean_text(persona.get("uuid"))
        occupation = _clean_text(persona.get("occupation"))
        if not occupation:
            raise ValueError(f"직업 정보가 없는 페르소나입니다: {persona_id}")
        if PUBLIC_SERVANT_KEYWORD in occupation:
            raise ValueError(
                f"공무원 페르소나는 시민으로 선택할 수 없습니다: {persona_id}"
            )

        match = persona_selection_match(
            persona,
            region_scope=region_scope,
            province=province,
            district=district,
            age_min=age_min,
            age_max=age_max,
        )
        if not match["region_match"]:
            raise ValueError(f"정책 적용 지역과 다른 페르소나입니다: {persona_id}")
        if match["age_cohort"] is None:
            raise ValueError(
                f"정책 대상 또는 연령 경계선에 속하지 않는 페르소나입니다: {persona_id}"
            )
        evidence.append({"persona_id": persona_id, **match})
    return evidence
