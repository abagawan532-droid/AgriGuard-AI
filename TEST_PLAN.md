# Test plan

## Manual functional checks
1. Start with `python app.py`; home page should load.
2. Switch language to Kannada and Hindi; navigation and key UI labels should change.
3. Open Crop Disease Scanner; upload a valid JPG; with no model installed, verify that the app reports the missing model rather than inventing a prediction.
4. Upload a `.txt` file renamed as a non-image extension; verify rejection.
5. Upload an image larger than 12 MB; verify the size error.
6. Open Tree Scanner; verify missing-model behavior.
7. Search `mango`, `Mangifera indica`, `Ficus`, and a nonexistent species in the reference library.
8. Open each tree profile and inspect all displayed fields.
9. Calculator: price 100, quantity 12, costs 200 → gross 1200, net 1000.
10. Calculator: negative value → validation/error message.
11. Check layout at desktop and mobile widths.

## Model evaluation before demonstration
- Keep independent train/validation/test partitions.
- Report per-class precision, recall, F1 and confusion matrix.
- Test images from different phones, backgrounds, lighting, leaf ages and regions.
- Test look-alike classes and unknown species.
- Never claim accuracy based only on training results.
