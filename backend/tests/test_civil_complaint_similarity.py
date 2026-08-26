import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.ai_simulation_core.complaints import (
    civil_complaint_similarity as similarity,
)
from backend.ai_simulation_core.complaints.civil_complaint_corpus import (
    FAQ_DATA_DIR,
    civil_complaint_source_fingerprint,
    load_civil_complaint_corpus,
)
from scripts.rag import build_civil_complaint_index as build_index


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, documents, **kwargs):
        self.calls.append(list(documents))
        rows = []
        for index, _ in enumerate(documents):
            rows.append([1.0, float(index + 1), 0.5])
        return np.asarray(rows, dtype=np.float32)


class FakeCollection:
    def __init__(
        self,
        *,
        count: int,
        ids: list[str],
        distances_by_row: list[list[float]] | None = None,
    ) -> None:
        self._count = count
        self.ids = ids
        self.distances_by_row = distances_by_row
        self.query_calls: list[dict] = []

    def count(self) -> int:
        return self._count

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        row_count = len(kwargs["query_embeddings"])
        if self.distances_by_row is None:
            distances = [[0.1] * len(self.ids) for _ in range(row_count)]
        else:
            if len(self.distances_by_row) != row_count:
                raise AssertionError(
                    "테스트 거리 행 수가 query embedding 수와 다릅니다."
                )
            distances = self.distances_by_row
        return {
            "ids": [list(self.ids) for _ in range(row_count)],
            "distances": distances,
        }


def _detail(
    case_id: str,
    *,
    title: str,
    question: str,
    answer: str,
    organization: str,
) -> dict:
    return {
        "faqNo": case_id,
        "dutySctnNm": "tqapttn",
        "qnaTitl": title,
        "qstnCntnCl": question,
        "ansCntnCl": answer,
        "ancName": organization,
        "deptName": "정책과",
        "regDate": "20260825",
        "ancCode": "100",
        "deptCode": "101",
        "lawList": [
            {
                "fullName": "주거기본법 / 제1조(목적)",
                "lwrdNm": "주거기본법",
                "lwrdUrl": "https://law.go.kr/example",
            }
        ],
        "subjList": [],
    }


def _write_data(data_dir: Path, details: list[dict]) -> None:
    rows = [
        {
            "faqNo": detail["faqNo"],
            "list": {"faqNo": detail["faqNo"], "title": detail["qnaTitl"]},
            "detail": detail,
            "matchedPolicy": {"서비스명": "검색 gate에 쓰면 안 되는 정책"},
            "matchedKeyword": "금지",
        }
        for detail in details
    ]
    (data_dir / "civil_policy_qna_detail.json").write_text(
        json.dumps({"data": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "civil_policy_qna_metadata.json").write_text(
        json.dumps({"savedDetailCount": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _manifest(data_dir: Path) -> dict:
    return {
        "schema_version": similarity.MANIFEST_SCHEMA_VERSION,
        "collection_name": similarity.DEFAULT_COLLECTION_NAME,
        "embedding_model": similarity.DEFAULT_MODEL,
        **civil_complaint_source_fingerprint(data_dir),
        "built_at": "2026-08-25T00:00:00+00:00",
        "source": {
            "source_kind": "public_faq_snapshot",
            "matched_policy_used": False,
        },
    }


def _version_manifest(built_at: str) -> dict:
    return {
        "schema_version": similarity.MANIFEST_SCHEMA_VERSION,
        "collection_name": similarity.DEFAULT_COLLECTION_NAME,
        "embedding_model": similarity.DEFAULT_MODEL,
        "embedding_dimension": 3,
        "detail_sha256": "d" * 64,
        "metadata_sha256": "m" * 64,
        "raw_record_count": 1,
        "unique_count": 1,
        "built_at": built_at,
        "source": {
            "source_kind": "public_faq_snapshot",
            "matched_policy_used": False,
        },
        "activation": {
            "kind": "atomic_pointer",
            "logical_path": "data/indexes/civil_complaints/current",
            "pointer_file": similarity.ACTIVE_POINTER_FILENAME,
            "reload_strategy": similarity.ACTIVE_RELOAD_STRATEGY,
            "restart_required": False,
        },
    }


def _write_version_manifest(version_dir: Path, built_at: str) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "manifest.json").write_text(
        json.dumps(_version_manifest(built_at), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _activate_pointer(logical_dir: Path, version_dir: Path) -> None:
    manifest_hash = hashlib.sha256(
        (version_dir / "manifest.json").read_bytes()
    ).hexdigest()
    build_index._write_active_pointer(
        logical_dir,
        {
            "schema_version": similarity.ACTIVE_POINTER_SCHEMA_VERSION,
            "active_version": version_dir.name,
            "version_path": os.path.relpath(version_dir, start=logical_dir),
            "manifest_sha256": manifest_hash,
            "activated_at": "2026-08-25T00:00:00+00:00",
            "reload_strategy": similarity.ACTIVE_RELOAD_STRATEGY,
            "restart_required": False,
        },
    )


def _specific_policy() -> dict:
    return {
        "policy_name": "평택시 청년 월세 지원",
        "target_audience": "평택시 거주 만 19세 이상 34세 이하 청년",
        "benefits": "월세 월 20만 원 지원",
        "region_scope": "specific",
        "region_province": "경기",
        "region_district": "경기-평택시",
        "age_min": 19,
        "age_max": 34,
    }


def _query_item(*, policy: dict | None = None, age: int = 29) -> dict:
    return {
        "complaint_text": (
            "청년 월세 지원 대상 자격에서 나이 때문에 제외되는 이유가 무엇인가요?"
        ),
        "policy": policy or _specific_policy(),
        "persona": {
            "age": age,
            "province": "경기",
            "district": "경기-평택시",
            "occupation": "무직",
        },
    }


def _relevant_detail(
    case_id: str = "housing-1", organization: str = "경기도 평택시"
) -> dict:
    return _detail(
        case_id,
        title="청년 월세 지원 대상 자격",
        question="만 19세 이상 34세 이하 청년만 지원 대상인가요?",
        answer="신청일 기준 만 19세 이상 34세 이하 청년이 지원 대상입니다.",
        organization=organization,
    )


def _service(
    data_dir: Path,
    *,
    collection: FakeCollection,
    embedder: FakeEmbedder | None = None,
    manifest: dict | None = None,
) -> similarity.CivilComplaintSimilarityService:
    return similarity.CivilComplaintSimilarityService(
        data_dir=data_dir,
        index_dir=data_dir / "unused-index",
        collection=collection,
        embedder=embedder or FakeEmbedder(),
        corpus=load_civil_complaint_corpus(data_dir),
        manifest=manifest or _manifest(data_dir),
    )


class CivilComplaintSimilarityTest(unittest.TestCase):
    def test_screenshot_false_positive_fails_all_relevant_hard_gates(self) -> None:
        corpus = load_civil_complaint_corpus(FAQ_DATA_DIR)
        candidate = next(record for record in corpus if record["case_id"] == "6910486")
        collection = FakeCollection(
            count=len(corpus),
            ids=[candidate["case_id"]],
        )
        service = similarity.CivilComplaintSimilarityService(
            index_dir=FAQ_DATA_DIR / "unused-index",
            data_dir=FAQ_DATA_DIR,
            collection=collection,
            embedder=FakeEmbedder(),
            corpus=corpus,
            manifest={
                "schema_version": similarity.MANIFEST_SCHEMA_VERSION,
                "collection_name": similarity.DEFAULT_COLLECTION_NAME,
                "embedding_model": similarity.DEFAULT_MODEL,
                **civil_complaint_source_fingerprint(FAQ_DATA_DIR),
                "built_at": "2026-08-25T00:00:00+00:00",
                "source": {
                    "source_kind": "public_faq_snapshot",
                    "matched_policy_used": False,
                },
            },
        )

        response = service.search_batch([_query_item()])[0]

        self.assertEqual(response["status"], "no_reliable_match")
        self.assertEqual(response["results"], [])
        self.assertGreaterEqual(response["rejection_counts"]["region"], 1)
        self.assertGreaterEqual(response["rejection_counts"]["domain"], 1)
        self.assertGreaterEqual(response["rejection_counts"]["issue"], 1)
        self.assertGreaterEqual(response["rejection_counts"]["qualification"], 1)

    def test_specific_region_rejects_unknown_candidate_organization(self) -> None:
        detail = _relevant_detail(organization="행복지원센터")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
            )
            response = service.search_batch([_query_item()])[0]

        self.assertEqual(response["status"], "no_reliable_match")
        self.assertEqual(response["rejection_counts"]["region"], 1)

    def test_nationwide_policy_allows_candidate_from_another_region(self) -> None:
        detail = _relevant_detail(organization="대전광역시")
        policy = {
            **_specific_policy(),
            "policy_name": "전국 청년 월세 지원",
            "region_scope": "nationwide",
            "region_province": "",
            "region_district": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
            )
            response = service.search_batch([_query_item(policy=policy)])[0]

        self.assertEqual(response["status"], "matched")
        self.assertTrue(response["results"][0]["reference_eligible"])
        self.assertEqual(
            response["results"][0]["evidence"]["region"]["reason"],
            "nationwide_policy",
        )

    def test_explicit_unrestricted_age_contract_is_not_an_invalid_query(self) -> None:
        detail = _relevant_detail(organization="대전광역시")
        policy = {
            **_specific_policy(),
            "policy_name": "전국 월세 지원",
            "target_audience": "전국 모든 연령의 시민",
            "region_scope": "nationwide",
            "region_province": "",
            "region_district": "",
            "age_min": None,
            "age_max": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
            )
            query, errors = service._prepare_query(
                _query_item(policy=policy, age=89)
            )

        self.assertEqual(errors, [])
        self.assertIsNotNone(query)
        self.assertFalse(query["age_active"])
        self.assertIsNone(query["age_range"])

    def test_english_policy_terms_identify_supported_domains(self) -> None:
        cases = {
            "Youth housing support for homeowners and renters": "housing",
            "Career and employment support": "employment",
            "University tuition scholarship": "education",
            "Medical treatment at a hospital": "health",
            "Family childcare and caregiving": "family_care",
            "Agriculture and fishery support": "agriculture_fisheries",
            "Public transport and railway pass": "transport",
            "Small business loan and debt relief": "business_finance",
            "Local tax payment": "tax",
            "Waste and water quality program": "environment",
            "Culture, sports, and library access": "culture_sports",
            "Permit and certificate registration": "legal_administration",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIn(expected, similarity.domain_tags(text))

    def test_age_35_boundary_matches_same_19_to_34_rule(self) -> None:
        detail = _relevant_detail()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
            )
            response = service.search_batch([_query_item(age=35)])[0]

        self.assertEqual(response["status"], "matched")
        evidence = response["results"][0]["evidence"]["age"]
        self.assertEqual(evidence["persona_relationship"], "upper_boundary")
        self.assertEqual(
            evidence["matched_candidate_range"],
            {"minimum": 19, "maximum": 34},
        )

    def test_batch_uses_one_encode_and_one_chroma_query(self) -> None:
        detail = _relevant_detail()
        embedder = FakeEmbedder()
        collection = FakeCollection(count=1, ids=[detail["faqNo"]])
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=collection,
                embedder=embedder,
            )
            responses = service.search_batch([_query_item(age=29), _query_item(age=35)])

        self.assertEqual([item["status"] for item in responses], ["matched", "matched"])
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(len(embedder.calls[0]), 4)
        self.assertEqual(len(collection.query_calls), 1)
        self.assertEqual(len(collection.query_calls[0]["query_embeddings"]), 4)

    def test_manifest_hash_change_fails_closed_before_query(self) -> None:
        detail = _relevant_detail()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            manifest = _manifest(data_dir)
            (data_dir / "civil_policy_qna_metadata.json").write_text(
                json.dumps({"savedDetailCount": 999}),
                encoding="utf-8",
            )
            collection = FakeCollection(count=1, ids=[detail["faqNo"]])
            service = _service(
                data_dir,
                collection=collection,
                manifest=manifest,
            )
            with self.assertRaises(similarity.CivilComplaintIndexUnavailableError):
                service.search_batch([_query_item()])

        self.assertEqual(collection.query_calls, [])

    def test_manifest_count_change_fails_closed(self) -> None:
        detail = _relevant_detail()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            manifest = _manifest(data_dir)
            manifest["unique_count"] = 2
            service = _service(
                data_dir,
                collection=FakeCollection(count=2, ids=[detail["faqNo"]]),
                manifest=manifest,
            )
            with self.assertRaises(similarity.CivilComplaintIndexUnavailableError):
                service.search_batch([_query_item()])

    def test_ui_threshold_is_inclusive_and_next_lower_score_is_hidden(self) -> None:
        detail = _relevant_detail()
        # Dense=.70 => semantic=.70 and all hard-gate context=.1 contribution.
        # lexical=.45 therefore makes final exactly .68.
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(
                    count=1,
                    ids=[detail["faqNo"]],
                    distances_by_row=[[0.30], [0.30]],
                ),
            )
            with patch.object(similarity, "lexical_score", return_value=0.45):
                accepted = service.search_batch([_query_item()])[0]

            lower_service = _service(
                data_dir,
                collection=FakeCollection(
                    count=1,
                    ids=[detail["faqNo"]],
                    distances_by_row=[[0.30], [0.30]],
                ),
            )
            with patch.object(similarity, "lexical_score", return_value=0.4499):
                hidden = lower_service.search_batch([_query_item()])[0]

        self.assertEqual(accepted["status"], "matched")
        self.assertEqual(accepted["results"][0]["match_score"], 68.0)
        self.assertEqual(hidden["status"], "no_reliable_match")
        self.assertEqual(hidden["rejection_counts"]["below_ui_threshold"], 1)

    def test_dense_and_semantic_floors_are_inclusive(self) -> None:
        detail = _relevant_detail()
        cases = (
            # complaint=.50, policy=.90 -> semantic=.68
            ([[0.50], [0.10]], "matched", None),
            ([[0.5002], [0.10]], "no_reliable_match", "below_complaint_dense"),
            # complaint=.90, policy=.50 -> semantic=.72
            ([[0.10], [0.50]], "matched", None),
            ([[0.10], [0.5002]], "no_reliable_match", "below_policy_dense"),
            # Both=.55 -> semantic=.55 exactly.
            ([[0.45], [0.45]], "matched", None),
            ([[0.4502], [0.4502]], "no_reliable_match", "below_semantic"),
        )
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            for distances, expected_status, rejection_key in cases:
                with self.subTest(
                    distances=distances,
                    expected_status=expected_status,
                ):
                    service = _service(
                        data_dir,
                        collection=FakeCollection(
                            count=1,
                            ids=[detail["faqNo"]],
                            distances_by_row=distances,
                        ),
                    )
                    with patch.object(similarity, "lexical_score", return_value=1.0):
                        response = service.search_batch([_query_item()])[0]
                    self.assertEqual(response["status"], expected_status)
                    if rejection_key:
                        self.assertEqual(
                            response["rejection_counts"][rejection_key],
                            1,
                        )

    def test_initial_final_floor_is_inclusive_before_ui_gate(self) -> None:
        detail = _relevant_detail()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            exact_service = _service(
                data_dir,
                collection=FakeCollection(
                    count=1,
                    ids=[detail["faqNo"]],
                    distances_by_row=[[0.40], [0.40]],
                ),
            )
            # semantic=.60, context=1, lexical=.50 -> final=.62 exactly.
            with patch.object(similarity, "lexical_score", return_value=0.50):
                exact = exact_service.search_batch([_query_item()])[0]

            lower_service = _service(
                data_dir,
                collection=FakeCollection(
                    count=1,
                    ids=[detail["faqNo"]],
                    distances_by_row=[[0.40], [0.40]],
                ),
            )
            with patch.object(similarity, "lexical_score", return_value=0.499):
                lower = lower_service.search_batch([_query_item()])[0]

        self.assertEqual(exact["status"], "no_reliable_match")
        self.assertEqual(exact["rejection_counts"]["below_final"], 0)
        self.assertEqual(exact["rejection_counts"]["below_ui_threshold"], 1)
        self.assertEqual(lower["rejection_counts"]["below_final"], 1)

    def test_shared_threshold_constants_and_response_contract(self) -> None:
        self.assertEqual(similarity.COMPLAINT_DENSE_FLOOR, 0.50)
        self.assertEqual(similarity.POLICY_DENSE_FLOOR, 0.50)
        self.assertEqual(similarity.SEMANTIC_FLOOR, 0.55)
        self.assertEqual(similarity.FINAL_SCORE_FLOOR, 0.62)
        self.assertEqual(similarity.UI_REFERENCE_SCORE_FLOOR, 0.68)

        detail = _relevant_detail()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            response = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
            ).search_batch([_query_item()])[0]

        self.assertEqual(response["status"], "matched")
        self.assertEqual(response["source_count"], 1)
        self.assertFalse(response["source"]["matched_policy_used"])
        result = response["results"][0]
        self.assertEqual(result["source_kind"], "public_faq_snapshot")
        self.assertTrue(result["reference_eligible"])
        self.assertIn(result["confidence"], {"low", "medium"})
        self.assertNotEqual(result["confidence"], "high")
        self.assertEqual(result["match_score"], result["component_scores"]["final"])
        self.assertTrue(all("type" in reason for reason in result["match_reasons"]))

    def test_invalid_query_and_empty_batch_do_not_embed(self) -> None:
        detail = _relevant_detail()
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            _write_data(data_dir, [detail])
            service = _service(
                data_dir,
                collection=FakeCollection(count=1, ids=[detail["faqNo"]]),
                embedder=embedder,
            )
            self.assertEqual(service.search_batch([]), [])
            response = service.search_batch(
                [{"complaint_text": "", "policy": {}, "persona": {}}]
            )[0]

        self.assertEqual(response["status"], "invalid_query")
        self.assertEqual(response["results"], [])
        self.assertIn("complaint_text_required", response["validation_errors"])
        self.assertEqual(embedder.calls, [])


class CivilComplaintPointerActivationTest(unittest.TestCase):
    def test_activation_failure_keeps_previous_pointer_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical_dir = root / "current"
            old_version = root / "versions" / "old-version"
            _write_version_manifest(old_version, "2026-08-25T00:00:00+00:00")
            _activate_pointer(logical_dir, old_version)
            old_pointer = (logical_dir / "active.json").read_bytes()

            staging = root / ".staging-new"
            _write_version_manifest(staging, "2026-08-25T01:00:00+00:00")
            with (
                patch.object(build_index.os, "replace", side_effect=OSError("stop")),
                self.assertRaisesRegex(OSError, "stop"),
            ):
                build_index.activate_index(staging, logical_dir)

            self.assertEqual(
                (logical_dir / "active.json").read_bytes(),
                old_pointer,
            )
            resolved, _, _ = similarity.resolve_active_index_dir(logical_dir)
            self.assertEqual(resolved, old_version.resolve())

    def test_legacy_current_migration_preserves_all_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical_dir = root / "current"
            logical_dir.mkdir()
            legacy_marker = logical_dir / "chroma.sqlite3"
            legacy_marker.write_text("legacy-index", encoding="utf-8")
            resolved_before, digest_before, pointer_before = (
                similarity.resolve_active_index_dir(logical_dir)
            )
            self.assertEqual(resolved_before, logical_dir.resolve())
            self.assertIsNone(digest_before)
            self.assertIsNone(pointer_before)

            staging = root / ".staging-new"
            _write_version_manifest(staging, "2026-08-25T02:00:00+00:00")
            version_dir = build_index.activate_index(staging, logical_dir)

            self.assertEqual(legacy_marker.read_text(encoding="utf-8"), "legacy-index")
            self.assertTrue((logical_dir / "active.json").is_file())
            resolved_after, digest_after, pointer_after = (
                similarity.resolve_active_index_dir(logical_dir)
            )
            self.assertEqual(resolved_after, version_dir.resolve())
            self.assertIsNotNone(digest_after)
            self.assertEqual(pointer_after["restart_required"], False)

    def test_singleton_service_refreshes_manifest_and_version_bound_caches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical_dir = root / "current"
            first_version = root / "versions" / "version-one"
            second_version = root / "versions" / "version-two"
            _write_version_manifest(first_version, "2026-08-25T03:00:00+00:00")
            _write_version_manifest(second_version, "2026-08-25T04:00:00+00:00")
            _activate_pointer(logical_dir, first_version)

            service = similarity.CivilComplaintSimilarityService(index_dir=logical_dir)
            self.assertEqual(
                service.manifest["built_at"],
                "2026-08-25T03:00:00+00:00",
            )
            service._collection = object()
            service._corpus = ({"case_id": "stale"},)
            service._corpus_by_id = {"stale": {"case_id": "stale"}}
            service._source_verified = True
            service._collection_verified = True

            _activate_pointer(logical_dir, second_version)

            self.assertEqual(
                service.manifest["built_at"],
                "2026-08-25T04:00:00+00:00",
            )
            self.assertEqual(service.active_index_dir, second_version.resolve())
            self.assertIsNone(service._collection)
            self.assertIsNone(service._corpus)
            self.assertIsNone(service._corpus_by_id)
            self.assertFalse(service._source_verified)
            self.assertFalse(service._collection_verified)

    def test_staging_validation_checks_model_dimension_and_content_hash(self) -> None:
        class StagingCollection:
            def __init__(self) -> None:
                self.metadata = {
                    "hnsw:space": "cosine",
                    "schema_version": "1",
                    "embedding_model": similarity.DEFAULT_MODEL,
                    "source_kind": "public_faq_snapshot",
                }
                document = "제목: 청년 월세"
                self.snapshot = {
                    "ids": ["case-1"],
                    "documents": [document],
                    "metadatas": [
                        {
                            "case_id": "case-1",
                            "source_kind": "public_faq_snapshot",
                            "content_hash": build_index.content_hash(document),
                        }
                    ],
                    "embeddings": [[0.1, 0.2, 0.3]],
                }

            def count(self) -> int:
                return 1

            def get(self, *, include):
                self.requested_include = include
                return self.snapshot

        class StagingClient:
            def __init__(self, collection) -> None:
                self.collection = collection

            def get_collection(self, name):
                self.requested_name = name
                return self.collection

        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            manifest = _version_manifest("2026-08-25T05:00:00+00:00")
            (staging / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            collection = StagingCollection()
            client = StagingClient(collection)
            fake_chromadb = SimpleNamespace(
                PersistentClient=lambda path: client,
            )
            fingerprint = {
                key: manifest[key]
                for key in (
                    "detail_sha256",
                    "metadata_sha256",
                    "raw_record_count",
                    "unique_count",
                )
            }
            with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
                validated = build_index._validate_staging(
                    staging,
                    expected_ids={"case-1"},
                    expected_fingerprint=fingerprint,
                )
                self.assertEqual(validated["embedding_dimension"], 3)

                collection.metadata["embedding_model"] = "wrong-model"
                with self.assertRaisesRegex(RuntimeError, "embedding_model"):
                    build_index._validate_staging(
                        staging,
                        expected_ids={"case-1"},
                        expected_fingerprint=fingerprint,
                    )
                collection.metadata["embedding_model"] = similarity.DEFAULT_MODEL

                collection.snapshot["metadatas"][0]["content_hash"] = ""
                with self.assertRaisesRegex(RuntimeError, "content_hash"):
                    build_index._validate_staging(
                        staging,
                        expected_ids={"case-1"},
                        expected_fingerprint=fingerprint,
                    )
                collection.snapshot["metadatas"][0]["content_hash"] = (
                    build_index.content_hash(collection.snapshot["documents"][0])
                )

                collection.snapshot["embeddings"][0] = [0.1, 0.2]
                with self.assertRaisesRegex(RuntimeError, "embedding dimension"):
                    build_index._validate_staging(
                        staging,
                        expected_ids={"case-1"},
                        expected_fingerprint=fingerprint,
                    )


if __name__ == "__main__":
    unittest.main()
