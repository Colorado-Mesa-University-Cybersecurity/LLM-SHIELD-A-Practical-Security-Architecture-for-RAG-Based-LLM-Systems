# Environments Overview

This repository implements a secure Retrieval-Augmented Generation (RAG) architecture designed for both internal staff and external clients, as described in our paper:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

Two environments are provided in the `src` directory, each with distinct access controls, security policies, and use-cases:

## 1. customer-facing

This folder contains the client-facing environment. It is designed for use by external clients (such as customers or business partners) who require information retrieval from company sources. Security in this environment emphasizes strict access boundaries—clients can only interact with context and data relevant to them. Defensive system prompts, PII redaction, and input validation are tailored to prevent cross-client data leakage and block adversarial requests.

- **Audience:** External clients
- **Features:** Strict data isolation, enhanced prompt guard, output PII redaction
- **Example use:** Customer portals, knowledge bases

## 2. employee-facing

This folder contains the environment for internal company use. It is intended for employees and staff, offering broader document access and administrative tools. While security remains a priority (identity-based access control, PII redaction, prompt guard), permission policies and context retrieval are designed for internal collaboration and company operations. 

- **Audience:** Company employees
- **Features:** Employee authentication, role-based access, document management
- **Example use:** Internal dashboards, support tools, HR document access

## 3. documents for both environments

For the customer-facing environment, documents must be provided manually—users upload or supply the files needed for retrieval. In contrast, the internal-tools (employee-facing) environment is configured to download source documents directly from our Supabase (Postgres) database. However, since the company database is not public, all necessary files for both environments are already included in this repository, organized in the appropriate folders. This allows you to test and evaluate both environments without external data access.

---

## Architecture Summary

Both environments implement the same RAG security pipeline:

- **Prompt Guard:** Input validation and sanitization using Llama Prompt Guard.
- **PII Redaction:** Automated removal of sensitive identifiers at ingestion, in metadata, and output.
- **Identity-Based Access Control:** Access policies enforced by user role and authentication status.

---

*For more background and security analysis, see the full research paper and references in the repository.*