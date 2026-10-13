"""Exercise actual module initialization and signed sessions in isolated processes."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VARIABLE = "POINTCLOUD_WEB_SECRET_KEY"


def run_web_import(value, code="import media_web_app"):
    environment = dict(os.environ)
    environment.pop(VARIABLE, None)
    if value is not None:
        environment[VARIABLE] = value
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("value", [None, "", " " * 40, "test-only-short-value"])
def test_missing_or_invalid_secret_fails_without_disclosing_value(value):
    result = run_web_import(value)
    assert result.returncode != 0
    assert "RuntimeError: Set POINTCLOUD_WEB_SECRET_KEY" in result.stderr
    if value and value.strip():
        assert value not in result.stdout + result.stderr


def test_environment_secret_signs_sessions_and_rejects_tampering():
    # Deliberately public deterministic fixture, never a real credential.
    result = run_web_import("unit-test-only-not-a-real-secret!", """
import os
from itsdangerous import BadSignature
from media_web_app import app
assert app.secret_key == os.environ['POINTCLOUD_WEB_SECRET_KEY']
serializer = app.session_interface.get_signing_serializer(app)
assert serializer is not None
cookie = serializer.dumps({'probe': 'signed'})
assert serializer.loads(cookie) == {'probe': 'signed'}
try:
    serializer.loads('x' + cookie)
except BadSignature:
    pass
else:
    raise AssertionError('tampered cookie accepted')
with app.test_client() as client:
    with client.session_transaction() as session:
        session['probe'] = 'signed'
    with client.session_transaction() as session:
        assert session['probe'] == 'signed'
print('CONFIG_OK')
""")
    assert result.returncode == 0, "Configured web import or session check failed"
    assert result.stdout.strip() == "CONFIG_OK"
    assert "unit-test-only-not-a-real-secret!" not in result.stderr
