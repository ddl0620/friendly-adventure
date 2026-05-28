# Data Collection

Pipeline that produces `merged_dataset.json` (3,000 records: 1,500 benign + 1,500 malicious).

## Sources

- **Malicious:** PyPI Malregistry — https://github.com/lxyeternal/pypi_malregistry (~9,500 packages)
- **Benign:** PyPI live index — downloaded by `fetchpypi.py`

## Run order

```bash
pip install -r requirements.txt

# malicious
git clone https://github.com/lxyeternal/pypi_malregistry
python collect_malicious.py            # pypi_malregistry/ -> malicious.json

# benign
python fetchpypi.py                    # PyPI -> BenignSet/
python collect_benign.py               # BenignSet/  -> benign.json

# merge
python sample_data.py -n 1500 --buckets 20 50 100 200 \
    --benign benign.json --malicious malicious.json --output merged_dataset.json
```

## Files

- `fetchpypi.py` - download benign sdists from PyPI into `BenignSet/`
- `collect_malicious.py` - `pypi_malregistry/` -> `malicious.json`
- `collect_benign.py` - `BenignSet/` -> `benign.json`
- `sample_data.py` - length-matched balanced merge -> `merged_dataset.json`

Note: `fetchpypi.py` pulls from live PyPI (top + random), so the exact set is not bit-reproducible; `benign.json` holds the snapshot used in the paper.
