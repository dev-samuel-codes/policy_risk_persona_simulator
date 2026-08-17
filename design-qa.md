# Design QA: 시민 말풍선 및 정책 비교 패널

- source visual truth: `/workspace/samuel/policy_risk_persona_simulator/tmp/codex-qa/20260817-dialogue-policy-panel/reference-dialogue-bubble.png`
- implementation screenshot: `/workspace/samuel/policy_risk_persona_simulator/tmp/codex-qa/20260817-dialogue-policy-panel/result-desktop.png`
- focused comparison: `/workspace/samuel/policy_risk_persona_simulator/tmp/codex-qa/20260817-dialogue-policy-panel/dialogue-comparison.png`
- viewport: desktop 1440 x 1000 CSS px, mobile 390 x 844 CSS px
- source pixels: 468 x 132
- implementation pixels: 1440 x 1000; focused dialogue region 648 x 192
- density normalization: source was scaled to 192 px height and compared beside the 648 x 192 implementation crop
- state: completed citizen simulation, first persona selected, similar-policy panel closed for primary comparison

## Full-view comparison evidence

The implemented result screen preserves the existing CivicEcho navigation and persona sidebar. The main content now has citizen dialogue on the left, a continuous vertical divider, and the entered-policy panel on the right. The similar-policy control is placed directly above the entered policy. No horizontal overflow was observed at 390 px (`scrollWidth = innerWidth = 390`).

## Focused comparison evidence

The combined comparison shows the required reference anatomy in both states: circular profile at left, large white rounded speech bubble at right, cool neutral background, generous inner padding, and a soft shadow. The implementation intentionally uses the CivicEcho profile fallback icon and real generated citizen dialogue rather than the reference robot artwork and sample greeting.

## Required fidelity surfaces

- Fonts and typography: existing Pretendard GOV font retained; 16–17 px dialogue type, 7–8 px effective line rhythm, and medium weight keep long Korean responses readable.
- Spacing and layout rhythm: 14–16 px avatar-to-bubble gap, 20–24 px bubble padding, 26 px bubble radius, and repeated 20 px conversation spacing match the reference's airy rhythm.
- Colors and tokens: white bubble, cool gray page surface, navy text/accent, subtle gray border, and low-opacity shadow remain consistent with both the reference mood and CivicEcho tokens.
- Image quality and asset fidelity: real persona photos render when present; otherwise the existing Lucide profile icon is used inside a crisp circular surface. The robot is treated as reference-only imagery, not copied into citizen profiles.
- Copy and content: generated citizen dialogue, complaint explanation, profile name, and entered-policy fields remain real application data. No mock copy is shipped in the UI.

## Findings

- No actionable P0, P1, or P2 differences remain.
- P3 intentional deviation: the reference robot avatar is replaced by the product's citizen profile/photo treatment so the component reflects the selected persona.
- P3 intentional deviation: long policy-response copy creates taller bubbles than the short reference greeting; the responsive layout keeps the same visual structure without truncating the citizen response.

## Interaction verification

- First persona result rendered with two speech bubbles.
- Selecting a second persona replaced the visible dialogue, and selecting the first persona restored it.
- `유사한 정책 찾아보기` changed `aria-expanded` from false to true and revealed real server results while keeping `입력된 정책` visible.
- Desktop and mobile first view rendered without horizontal overflow.
- Browser console warning/error list was empty.

## Comparison history

- Initial focused comparison: no P0/P1/P2 issue found; no visual repair loop required.

## Follow-up polish

- When persona image data becomes available, the circular fallback automatically changes to the real profile image.

final result: passed
