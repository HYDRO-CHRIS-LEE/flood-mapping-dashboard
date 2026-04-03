"""tests/test_integration.py — verify app serves frontend and API together."""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_index_html_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "EarthAI" in resp.text
    assert "sidebar-nav" in resp.text


def test_page_html_served():
    resp = client.get("/pages/rainfall.html")
    assert resp.status_code == 200
    assert "Rainfall Analysis" in resp.text


def test_api_and_frontend_coexist():
    api_resp = client.get("/api/events")
    assert api_resp.status_code == 200
    assert api_resp.json()["ok"] is True

    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "<!DOCTYPE html>" in html_resp.text


def test_js_files_served():
    resp = client.get("/js/app.js")
    assert resp.status_code == 200
    assert "PAGES" in resp.text

    resp = client.get("/js/api.js")
    assert resp.status_code == 200
    assert "API" in resp.text


def test_all_page_templates_exist():
    for page in ["rainfall", "optical", "sar", "classifier", "flappy"]:
        resp = client.get(f"/pages/{page}.html")
        assert resp.status_code == 200, f"Missing page: {page}"
        assert len(resp.text) > 50, f"Page too short: {page}"
