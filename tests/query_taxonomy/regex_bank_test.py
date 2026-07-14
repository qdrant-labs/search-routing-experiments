import pytest
from query_taxonomy.banks import HTTPStatusCodeBank, IdentifierMatch


@pytest.fixture
def status_code_bank():
    return HTTPStatusCodeBank()


def texts(matches: list[IdentifierMatch]) -> list[str]:
    return [m.text for m in matches]


# --- Keyword before code: (HTTP|error|status|code) <sep> <3-digit code> ---

def test_matches_http_keyword_then_code(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("HTTP 200")) == ['HTTP 200']

def test_matches_http_keyword_then_code_in_prose(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("We should send HTTP 200 code")) == ['HTTP 200']

def test_matches_http_with_non_space_separator(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("We should send HTTP - 200 code")) == ['HTTP - 200']

def test_matches_http_protocol_prefix(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("HTTP/1.1 201 CREATED")) == ['HTTP/1.1 201']

def test_matches_error_keyword_then_code(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("We should send error 500 code")) == ['error 500']


# --- Code before reason: <3-digit code> <reason word> ---

def test_matches_code_then_reason_and_code_then_informal_reason(status_code_bank: HTTPStatusCodeBank):
    assert texts(status_code_bank.matches("Expected to get 200 OK, got 500 error code")) == ['200 OK', '500 error']


# --- False positives ---

def test_does_not_match_bare_number_in_prose(status_code_bank: HTTPStatusCodeBank):
    assert status_code_bank.matches("There are bunch of errors created. 200 people died") == []
