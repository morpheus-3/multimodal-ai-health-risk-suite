# Verification results

Verified on 9 October 2026 with Python 3.10, Java 21 (operations only) and Chromium.

**4 automated Python/API tests passed.** Ensemble training and reload, held-out ROC-AUC, SHAP additivity, invalid inputs, and Groq API contract.

**Browser end-to-end scenario passed**, with desktop (1440 x 1000) and mobile (390 x 844) layouts. No JavaScript runtime errors or horizontal overflow were observed. Screenshots are included in `screenshots/`; `BROWSER_TEST_RESULTS.json` records the outcome.

Tests train local models on deterministic synthetic fixtures. These verify software behavior and serialization, not real-world predictive quality. Demo screenshots use the same explicitly labeled synthetic models. Groq request contracts are mocked in automated tests, and the browser scenario checks missing-key handling. Live Groq inference needs a valid configured key and quota and has not been exercised.

Run the test command in README.md to reproduce automated checks. The GitHub Actions workflow runs these tests independently on every push and pull request.
