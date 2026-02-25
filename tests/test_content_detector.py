"""Tests for content type detection module."""

import unittest

from src.test_analysis_assistant.content_detector import (
    CodeLanguage,
    ContentCategory,
    ContentDetectionResult,
    DocumentType,
    detect_code_language,
    detect_content_type,
    suggest_ingestion_strategy,
)


class TestContentDetectorPython(unittest.TestCase):
    """Test detection of Python code."""

    def test_detect_python_function(self):
        content = """def authenticate(user_id: str) -> bool:
    if not user_id:
        raise ValueError("user_id is required")
    return True
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.PYTHON)
        self.assertGreater(result.confidence, 0.1)
        self.assertEqual(result.recommended_chunker, "code_aware")

    def test_detect_python_imports(self):
        content = """import os
import sys
from typing import List, Dict, Optional

def process_data(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.PYTHON)
        self.assertGreater(result.confidence, 0.1)

    def test_detect_python_class(self):
        content = """class UserAuthenticator:
    def __init__(self, config: dict):
        self.config = config

    def authenticate(self, token: str) -> bool:
        return bool(token)
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.PYTHON)

    def test_detect_python_test_file(self):
        content = """
import pytest

def test_login_success():
    assert True

class TestAuth:
    def test_token_refresh(self):
        pass
"""
        result = detect_content_type(content, source_hint="tests/test_auth.py")

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.recommended_source_type, "code_snippet")


class TestContentDetectorJavaScript(unittest.TestCase):
    """Test detection of JavaScript/TypeScript code."""

    def test_detect_javascript(self):
        content = """
const fetchData = async (url) => {
    const response = await fetch(url);
    return response.json();
};

module.exports = { fetchData };
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.JAVASCRIPT)
        self.assertEqual(result.recommended_chunker, "code_aware")

    def test_detect_typescript(self):
        content = """
interface User {
    id: string;
    name: string;
    email: string;
}

function getUser(id: string): Promise<User> {
    return fetch(`/api/users/${id}`).then(r => r.json());
}
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.TYPESCRIPT)


class TestContentDetectorGo(unittest.TestCase):
    """Test detection of Go code."""

    def test_detect_go(self):
        content = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, world!")
}

func processData(data string) error {
    if data == "" {
        return fmt.Errorf("data is empty")
    }
    return nil
}
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.GO)


class TestContentDetectorJava(unittest.TestCase):
    """Test detection of Java code."""

    def test_detect_java(self):
        content = """
package com.example.auth;

import java.util.Optional;

public class UserService {
    public Optional<User> findById(String id) {
        return Optional.empty();
    }
}
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.CODE)
        self.assertEqual(result.detected_language, CodeLanguage.JAVA)


class TestContentDetectorMarkdown(unittest.TestCase):
    """Test detection of markdown content."""

    def test_detect_markdown(self):
        content = """# Authentication System

This document describes the authentication system.

## Features

- Login
- Logout
- Token refresh
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.MARKDOWN)
        self.assertGreater(result.confidence, 0.3)

    def test_detect_readme(self):
        content = """
# My Project

## Installation

Run `npm install` to install dependencies.

## Usage

```bash
npm start
```
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.MARKDOWN)
        self.assertEqual(result.detected_doc_type, DocumentType.README)


class TestContentDetectorRequirements(unittest.TestCase):
    """Test detection of requirements documents."""

    def test_detect_requirements(self):
        content = """
# Requirements Document

## Functional Requirements

The system shall authenticate users using JWT tokens.

### FR-001: User Login

The system must accept valid credentials and return an access token.

### FR-002: Token Refresh

The system should allow refreshing expired tokens.

## Acceptance Criteria

1. Login returns JWT token
2. Token expires after 1 hour
"""
        result = detect_content_type(content)

        self.assertEqual(result.detected_doc_type, DocumentType.REQUIREMENTS)
        self.assertEqual(result.recommended_source_type, "requirements")

    def test_detect_user_story(self):
        content = """
User Story: Authentication

As a user,
I want to log in with my credentials,
So that I can access protected resources.

Acceptance Criteria:
- Valid credentials return 200 OK
- Invalid credentials return 401 Unauthorized
"""
        result = detect_content_type(content)

        self.assertEqual(result.detected_doc_type, DocumentType.REQUIREMENTS)


class TestContentDetectorArchitecture(unittest.TestCase):
    """Test detection of architecture documents."""

    def test_detect_architecture(self):
        content = """
# System Architecture

## Overview

This document describes the system architecture for the authentication service.

## Components

### Microservices

- User Service
- Token Service
- Session Manager

### Data Flow

1. User sends credentials
2. Auth service validates
3. JWT token issued
"""
        result = detect_content_type(content)

        self.assertEqual(result.detected_doc_type, DocumentType.ARCHITECTURE)


class TestContentDetectorIncidentReport(unittest.TestCase):
    """Test detection of incident reports."""

    def test_detect_incident_report(self):
        content = """
# Incident Report: Auth Service Outage

## Summary

Auth service experienced downtime on 2024-01-15.

## Root Cause

Database connection pool exhaustion.

## Timeline

- 10:00 - Incident started
- 10:15 - Alert triggered
- 10:30 - Mitigation applied

## Impact

5000 users affected for 30 minutes.
"""
        result = detect_content_type(content)

        self.assertEqual(result.detected_doc_type, DocumentType.INCIDENT_REPORT)
        self.assertEqual(result.recommended_source_type, "system_analysis")


class TestContentDetectorJSON(unittest.TestCase):
    """Test detection of JSON content."""

    def test_detect_json(self):
        content = """
{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "lodash": "^4.17.0"
    }
}
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.STRUCTURED_DATA)
        self.assertEqual(result.detected_language, CodeLanguage.JSON)


class TestContentDetectorYAML(unittest.TestCase):
    """Test detection of YAML content."""

    def test_detect_yaml(self):
        content = """services:
  - name: auth
    port: 8080
    replicas: 3

database:
  host: localhost
  port: 5432
"""
        result = detect_content_type(content)

        self.assertEqual(result.category, ContentCategory.STRUCTURED_DATA)
        self.assertEqual(result.detected_language, CodeLanguage.YAML)


class TestSuggestIngestionStrategy(unittest.TestCase):
    """Test ingestion strategy suggestions."""

    def test_suggest_code_strategy(self):
        content = """def hello():
    print("Hello, World!")
"""
        chunker, source_type = suggest_ingestion_strategy(content)

        self.assertEqual(chunker, "code_aware")
        self.assertEqual(source_type, "repository")

    def test_suggest_requirements_strategy(self):
        content = """
The system shall authenticate users.

User story: Login functionality
"""
        chunker, source_type = suggest_ingestion_strategy(content)

        self.assertEqual(source_type, "requirements")

    def test_suggest_system_analysis_strategy(self):
        content = """
Root cause analysis for auth failure.

Architecture diagram:
"""
        chunker, source_type = suggest_ingestion_strategy(content)

        self.assertEqual(source_type, "system_analysis")


class TestDetectCodeLanguage(unittest.TestCase):
    """Test standalone code language detection."""

    def test_detect_rust(self):
        content = """
fn main() {
    println!("Hello, world!");
}

pub fn process_data(data: &str) -> Result<(), Error> {
    Ok(())
}
"""
        lang = detect_code_language(content)

        self.assertEqual(lang, CodeLanguage.RUST)

    def test_detect_sql(self):
        content = """SELECT users.name, COUNT(orders.id) as order_count
FROM users
LEFT JOIN orders ON users.id = orders.user_id
WHERE users.active = true
GROUP BY users.name
"""
        lang = detect_code_language(content)

        self.assertEqual(lang, CodeLanguage.SQL)

    def test_detect_bash(self):
        content = """#!/bin/bash

set -e

echo "Building project..."
npm run build
"""
        lang = detect_code_language(content)

        self.assertEqual(lang, CodeLanguage.BASH)

    def test_non_code_returns_none(self):
        content = "This is just plain text content without any code."

        lang = detect_code_language(content)

        self.assertIsNone(lang)


class TestContentDetectionEdgeCases(unittest.TestCase):
    """Test edge cases in content detection."""

    def test_empty_content(self):
        result = detect_content_type("")

        self.assertEqual(result.category, ContentCategory.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_whitespace_only(self):
        result = detect_content_type("   \n\n   \t   ")

        self.assertEqual(result.category, ContentCategory.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_short_code_snippet(self):
        content = "x + 1"

        result = detect_content_type(content)

        # Short content might not have enough indicators
        self.assertIsNotNone(result.category)


if __name__ == "__main__":
    unittest.main()
