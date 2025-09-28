"""
Download the latest WA rental bonds ZIP from the AHDAP CKAN portal.

The dataset is identified by name via CKAN's `package_search`; the latest
`.zip` resource by last-modified/created time is downloaded into `data_raw/`.
"""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import certifi
import requests

from src.config import CKAN_BASE, CKAN_DATASET_NAME, RAW_DIR

CERTS_DIR = Path(__file__).resolve().parents[2] / "certs"
_SECTIGO_DV_R36_PEM = """-----BEGIN CERTIFICATE-----
MIIGTDCCBDSgAwIBAgIQOXpmzCdWNi4NqofKbqvjsTANBgkqhkiG9w0BAQwFADBf
MQswCQYDVQQGEwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQD
Ey1TZWN0aWdvIFB1YmxpYyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYw
HhcNMjEwMzIyMDAwMDAwWhcNMzYwMzIxMjM1OTU5WjBgMQswCQYDVQQGEwJHQjEY
MBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTcwNQYDVQQDEy5TZWN0aWdvIFB1Ymxp
YyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gQ0EgRFYgUjM2MIIBojANBgkqhkiG9w0B
AQEFAAOCAY8AMIIBigKCAYEAljZf2HIz7+SPUPQCQObZYcrxLTHYdf1ZtMRe7Yeq
RPSwygz16qJ9cAWtWNTcuICc++p8Dct7zNGxCpqmEtqifO7NvuB5dEVexXn9RFFH
12Hm+NtPRQgXIFjx6MSJcNWuVO3XGE57L1mHlcQYj+g4hny90aFh2SCZCDEVkAja
EMMfYPKuCjHuuF+bzHFb/9gV8P9+ekcHENF2nR1efGWSKwnfG5RawlkaQDpRtZTm
M64TIsv/r7cyFO4nSjs1jLdXYdz5q3a4L0NoabZfbdxVb+CUEHfB0bpulZQtH1Rv
38e/lIdP7OTTIlZh6OYL6NhxP8So0/sht/4J9mqIGxRFc0/pC8suja+wcIUna0HB
pXKfXTKpzgis+zmXDL06ASJf5E4A2/m+Hp6b84sfPAwQ766rI65mh50S0Di9E3Pn
2WcaJc+PILsBmYpgtmgWTR9eV9otfKRUBfzHUHcVgarub/XluEpRlTtZudU5xbFN
xx/DgMrXLUAPaI60fZ6wA+PTAgMBAAGjggGBMIIBfTAfBgNVHSMEGDAWgBRWc1hk
lfmSGrASKgRieaFAFYghSTAdBgNVHQ4EFgQUaMASFhgOr872h6YyV6NGUV3LBycw
DgYDVR0PAQH/BAQDAgGGMBIGA1UdEwEB/wQIMAYBAf8CAQAwHQYDVR0lBBYwFAYI
KwYBBQUHAwEGCCsGAQUFBwMCMBsGA1UdIAQUMBIwBgYEVR0gADAIBgZngQwBAgEw
VAYDVR0fBE0wSzBJoEegRYZDaHR0cDovL2NybC5zZWN0aWdvLmNvbS9TZWN0aWdv
UHVibGljU2VydmVyQXV0aGVudGljYXRpb25Sb290UjQ2LmNybDCBhAYIKwYBBQUH
AQEEeDB2ME8GCCsGAQUFBzAChkNodHRwOi8vY3J0LnNlY3RpZ28uY29tL1NlY3Rp
Z29QdWJsaWNTZXJ2ZXJBdXRoZW50aWNhdGlvblJvb3RSNDYucDdjMCMGCCsGAQUF
BzABhhdodHRwOi8vb2NzcC5zZWN0aWdvLmNvbTANBgkqhkiG9w0BAQwFAAOCAgEA
YtOC9Fy+TqECFw40IospI92kLGgoSZGPOSQXMBqmsGWZUQ7rux7cj1du6d9rD6C8
ze1B2eQjkrGkIL/OF1s7vSmgYVafsRoZd/IHUrkoQvX8FZwUsmPu7amgBfaY3g+d
q1x0jNGKb6I6Bzdl6LgMD9qxp+3i7GQOnd9J8LFSietY6Z4jUBzVoOoz8iAU84OF
h2HhAuiPw1ai0VnY38RTI+8kepGWVfGxfBWzwH9uIjeooIeaosVFvE8cmYUB4TSH
5dUyD0jHct2+8ceKEtIoFU/FfHq/mDaVnvcDCZXtIgitdMFQdMZaVehmObyhRdDD
4NQCs0gaI9AAgFj4L9QtkARzhQLNyRf87Kln+YU0lgCGr9HLg3rGO8q+Y4ppLsOd
unQZ6ZxPNGIfOApbPVf5hCe58EZwiWdHIMn9lPP6+F404y8NNugbQixBber+x536
WrZhFZLjEkhp7fFXf9r32rNPfb74X/U90Bdy4lzp3+X1ukh1BuMxA/EEhDoTOS3l
7ABvc7BYSQubQ2490OcdkIzUh3ZwDrakMVrbaTxUM2p24N6dB+ns2zptWCva6jzW
r8IWKIMxzxLPv5Kt3ePKcUdvkBU/smqujSczTzzSjIoR5QqQA6lN1ZRSnuHIWCvh
JEltkYnTAH41QJ6SAWO66GrrUESwN/cgZzL4JLEqz1Y=
-----END CERTIFICATE-----"""

_SECTIGO_ROOT_R46_PEM = """-----BEGIN CERTIFICATE-----
MIIFijCCA3KgAwIBAgIQdY39i658BwD6qSWn4cetFDANBgkqhkiG9w0BAQwFADBf
MQswCQYDVQQGEwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQD
Ey1TZWN0aWdvIFB1YmxpYyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYw
HhcNMjEwMzIyMDAwMDAwWhcNNDYwMzIxMjM1OTU5WjBfMQswCQYDVQQGEwJHQjEY
MBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQDEy1TZWN0aWdvIFB1Ymxp
YyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYwggIiMA0GCSqGSIb3DQEB
AQUAA4ICDwAwggIKAoICAQCTvtU2UnXYASOgHEdCSe5jtrch/cSV1UgrJnwUUxDa
ef0rty2k1Cz66jLdScK5vQ9IPXtamFSvnl0xdE8H/FAh3aTPaE8bEmNtJZlMKpnz
SDBh+oF8HqcIStw+KxwfGExxqjWMrfhu6DtK2eWUAtaJhBOqbchPM8xQljeSM9xf
iOefVNlI8JhD1mb9nxc4Q8UBUQvX4yMPFF1bFOdLvt30yNoDN9HWOaEhUTCDsG3X
ME6WW5HwcCSrv0WBZEMNvSE6Lzzpng3LILVCJ8zab5vuZDCQOc2TZYEhMbUjUDM3
IuM47fgxMMxF/mL50V0yeUKH32rMVhlATc6qu/m1dkmU8Sf4kaWD5QazYw6A3OAS
VYCmO2a0OYctyPDQ0RTp5A1NDvZdV3LFOxxHVp3i1fuBYYzMTYCQNFu31xR13NgE
SJ/AwSiItOkcyqex8Va3e0lMWeUgFaiEAin6OJRpmkkGj80feRQXEgyDet4fsZfu
+Zd4KKTIRJLpfSYFplhym3kT2BFfrsU4YjRosoYwjviQYZ4ybPUHNs2iTG7sijbt
8uaZFURww3y8nDnAtOFr94MlI1fZEoDlSfB1D++N6xybVCi0ITz8fAr/73trdf+L
HaAZBav6+CuBQug4urv7qv094PPK306Xlynt8xhW6aWWrL3DkJiy4Pmi1KZHQ3xt
zwIDAQABo0IwQDAdBgNVHQ4EFgQUVnNYZJX5khqwEioEYnmhQBWIIUkwDgYDVR0P
AQH/BAQDAgGGMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEMBQADggIBAC9c
mTz8Bl6MlC5w6tIyMY208FHVvArzZJ8HXtXBc2hkeqK5Duj5XYUtqDdFqij0lgVQ
YKlJfp/imTYpE0RHap1VIDzYm/EDMrraQKFz6oOht0SmDpkBm+S8f74TlH7Kph52
gDY9hAaLMyZlbcp+nv4fjFg4exqDsQ+8FxG75gbMY/qB8oFM2gsQa6H61SilzwZA
Fv97fRheORKkU55+MkIQpiGRqRxOF3yEvJ+M0ejf5lG5Nkc/kLnHvALcWxxPDkjB
JYOcCj+esQMzEhonrPcibCTRAUH4WAP+JWgiH5paPHxsnnVI84HxZmduTILA7rpX
DhjvLpr3Etiga+kFpaHpaPi8TD8SHkXoUsCjvxInebnMMTzD9joiFgOgyY9mpFui
TdaBJQbpdqQACj7LzTWb4OE4y2BThihCQRxEV+ioratF4yUQvNs+ZUH7G6aXD+u5
dHn5HrwdVw1Hr8Mvn4dGp+smWg9WY7ViYG4A++MnESLn/pmPNPW56MORcr3Ywx65
LvKRRFHQV80MNNVIIb/bE/FmJUNS0nAiNs2fxBx1IK1jcmMGDw4nztJqDby1ORrp
0XZ60Vzk50lJLVU3aPAaOpg+VBeHVOmmJ1CJeyAvP/+/oYtKR5j/K3tJPsMpRmAY
QqszKbrAKbkTidOIijlBO8n9pu0f9GBj39ItVQGL
-----END CERTIFICATE-----"""


def _extra_ca_files() -> List[Path]:
    """Ensure bundled Sectigo intermediates + root exist on disk and return their paths."""
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    files_and_contents = [
        (CERTS_DIR / "sectigo_public_server_auth_ca_dv_r36.pem", _SECTIGO_DV_R36_PEM),
        (CERTS_DIR / "sectigo_public_server_auth_root_r46.pem", _SECTIGO_ROOT_R46_PEM),
    ]
    created = []
    for path, pem in files_and_contents:
        if not path.exists():
            path.write_text(pem.strip() + "\n", encoding="ascii")
        created.append(path)
    return created
_COMBINED_CA_BUNDLE = CERTS_DIR / "ckan_ca_bundle.pem"


def _combine_ca_bundle(base_bundle: str, extras: List[Path]) -> str:
    """Append any extra CA certs to the default bundle and return the combined path."""
    available = [p for p in extras if p.exists()]
    if not available:
        return base_bundle

    base_path = Path(base_bundle)
    target = _COMBINED_CA_BUNDLE
    target.parent.mkdir(parents=True, exist_ok=True)

    base_text = base_path.read_text(encoding="ascii", errors="ignore").rstrip()
    extras_text = [p.read_text(encoding="ascii", errors="ignore").strip() for p in available]
    with target.open("w", encoding="ascii") as out:
        out.write(base_text)
        out.write("\n")
        for cert in extras_text:
            if cert:
                out.write("\n")
                out.write(cert)
                out.write("\n")

    return str(target)

API = f"{CKAN_BASE}/api/3/action"

def _build_session() -> requests.Session:
    """Configure a shared session with optional custom CA bundle or insecure mode."""
    verify_env = os.getenv("CKAN_VERIFY_SSL", "1").strip().lower()
    ca_bundle_env = os.getenv("CKAN_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE")
    verify_setting: bool | str

    if verify_env in {"0", "false", "no", "off"}:
        verify_setting = False
    else:
        base_bundle = certifi.where()
        if ca_bundle_env:
            base_bundle = os.path.expanduser(ca_bundle_env)
        verify_setting = _combine_ca_bundle(base_bundle, _extra_ca_files())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", verify_setting)
        os.environ.setdefault("SSL_CERT_FILE", verify_setting)

    session = requests.Session()
    session.verify = verify_setting
    trust_env_env = os.getenv("CKAN_TRUST_ENV", "0").strip().lower()
    session.trust_env = trust_env_env in {"1", "true", "yes", "on"}
    return session

SESSION = _build_session()

def _get(url: str, **params: Any) -> Dict[str, Any]:
    """GET a CKAN action endpoint and return parsed JSON (dict)."""
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def package_search(q: str) -> List[Dict[str, Any]]:
    """Basic search by dataset name; adjust if your CKAN uses different fields."""
    res = _get(f"{API}/package_search", q=q)
    return res["result"]["results"]

def latest_resource_url(pkg: Dict[str, Any]) -> str:
    """Choose the latest ZIP resource by `last_modified` or `created`."""
    resources = [
        r for r in pkg.get("resources", [])
        if (r.get("format", "") or "").lower() == "zip" or str(r.get("url", "")).lower().endswith(".zip")
    ]
    if not resources:
        raise RuntimeError("No ZIP resources found in dataset.")
    def key(r: Dict[str, Any]) -> str:
        return r.get("last_modified") or r.get("created") or ""
    resources.sort(key=key, reverse=True)
    return resources[0]["url"]

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION.verify is False:
        print("[warn] CKAN TLS verification disabled (CKAN_VERIFY_SSL=0). Proceed with caution.")
    elif isinstance(SESSION.verify, str):
        print(f"[info] Using custom CA bundle for CKAN requests: {SESSION.verify}")
    if not SESSION.trust_env:
        print("[info] Ignoring proxy env vars for CKAN (CKAN_TRUST_ENV=0)")
    pkgs = package_search(CKAN_DATASET_NAME)
    if not pkgs:
        raise SystemExit("Dataset not found. Check CKAN_DATASET_NAME in src/config.py.")
    url = latest_resource_url(pkgs[0])
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"wa_bonds_{ts}.zip"
    print(f"Downloading: {url}")
    with SESSION.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
