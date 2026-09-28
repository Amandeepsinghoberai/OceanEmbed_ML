import requests

# Pick one file that failed in the batch that had 100% failures
test_url = "https://data-argo.ifremer.fr/dac/aoml/1901517/profiles/D1901517_318.nc"

print(f"Testing: {test_url}")
try:
    r = requests.get(test_url, timeout=30)
    print(f"Status code: {r.status_code}")
    print(f"Response headers: {dict(r.headers)}")
    if r.status_code != 200:
        print(f"Response text (first 500 chars): {r.text[:500]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")