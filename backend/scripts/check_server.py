"""
Simple diagnostic script to check backend server health.
Usage: python scripts/check_server.py
"""

import sys
import httpx

BASE_URL = "http://localhost:8000"


def check_health():
    """Check if server is running and responding."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Health check: HTTP {resp.status_code}")
        print(f"  Response: {resp.json()}")
        return True
    except httpx.ConnectError:
        print(f"✗ Health check: Connection refused")
        print(f"  Server does not appear to be running at {BASE_URL}")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def check_upload_config():
    """Check upload configuration."""
    try:
        # Try to get upload endpoint info (will fail but shows server is responsive)
        resp = httpx.post(f"{BASE_URL}/api/videos/upload", timeout=2)
        # We expect 422 (validation error) if server is working
        if resp.status_code in (400, 422):
            print(f"✓ Upload endpoint is responsive (HTTP {resp.status_code} expected for empty request)")
            return True
        print(f"? Upload endpoint returned HTTP {resp.status_code}")
        return True
    except httpx.ConnectError:
        print(f"✗ Cannot connect to upload endpoint")
        return False
    except httpx.TimeoutException:
        print(f"✗ Upload endpoint timeout (2s)")
        return False
    except Exception as e:
        print(f"? Upload endpoint check: {e}")
        return True


def main():
    print("=" * 50)
    print("Backend Server Diagnostic")
    print("=" * 50)
    print()

    healthy = True

    print("1. Checking server health...")
    if not check_health():
        healthy = False
    print()

    print("2. Checking upload endpoint...")
    if not check_upload_config():
        healthy = False
    print()

    if healthy:
        print("=" * 50)
        print("✓ Server appears to be running normally")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("✗ Server issues detected")
        print()
        print("Troubleshooting steps:")
        print("1. Ensure uvicorn is running:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("2. Check for firewall/antivirus blocking port 8000")
        print("3. Try accessing http://localhost:8000/health in browser")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
