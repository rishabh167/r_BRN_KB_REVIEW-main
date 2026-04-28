-- ============================================================================
-- BRN KB Review Service - Database Schema
-- ============================================================================
-- Run this script to initialize the kb_reviews tables in the broadnetr database
-- ============================================================================
-- Main review table
CREATE TABLE IF NOT EXISTS kb_reviews (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  agent_id BIGINT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
  config_json LONGTEXT,
  total_documents INT DEFAULT 0,
  total_chunks INT DEFAULT 0,
  chunks_with_issues INT DEFAULT 0,
  candidate_pairs INT DEFAULT 0,
  url_duplicates_removed INT DEFAULT 0,
  issues_found INT DEFAULT 0,
  issues_resolved INT DEFAULT 0,
  progress INT DEFAULT 0,
  error_message LONGTEXT,
  started_at DATETIME,
  completed_at DATETIME,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  previous_review_id BIGINT,
  pairs_reused INT DEFAULT 0,
  pairs_analyzed INT DEFAULT 0,
  docs_changed INT DEFAULT 0,
  docs_unchanged INT DEFAULT 0,
  created_by_user_id BIGINT,
  INDEX idx_agent_id (agent_id),
  INDEX idx_status (status),
  INDEX idx_started_at (started_at)
);
-- Review issues found during analysis
CREATE TABLE IF NOT EXISTS kb_review_issues (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  review_id BIGINT NOT NULL,
  issue_type VARCHAR(30) NOT NULL,
  severity VARCHAR(10) NOT NULL,
  confidence FLOAT NOT NULL DEFAULT 1.0,
  title VARCHAR(255) NOT NULL,
  description LONGTEXT,
  doc_a_name VARCHAR(500),
  doc_a_page INT,
  doc_a_excerpt LONGTEXT,
  doc_b_name VARCHAR(500),
  doc_b_page INT,
  doc_b_excerpt LONGTEXT,
  entities_involved LONGTEXT,
  consensus VARCHAR(20),
  judges_flagged INT DEFAULT 0,
  judges_total INT DEFAULT 0,
  carried_forward BOOLEAN NOT NULL DEFAULT FALSE,
  original_review_id BIGINT,
  status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
  status_updated_by BIGINT,
  status_updated_at DATETIME,
  status_note LONGTEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_review_id (review_id),
  INDEX idx_status (status),
  FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE CASCADE
);
-- Judge verdicts for each issue
CREATE TABLE IF NOT EXISTS kb_review_judge_results (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  issue_id BIGINT NOT NULL,
  judge_index INT NOT NULL,
  judge_provider VARCHAR(30),
  judge_model VARCHAR(100),
  detected BOOLEAN NOT NULL DEFAULT FALSE,
  severity VARCHAR(10),
  confidence FLOAT,
  reasoning LONGTEXT,
  input_tokens INT DEFAULT 0,
  output_tokens INT DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_issue_id (issue_id),
  FOREIGN KEY (issue_id) REFERENCES kb_review_issues(id) ON DELETE CASCADE
);
-- Per-review judge stats (aggregate)
CREATE TABLE IF NOT EXISTS kb_review_judge_stats (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  review_id BIGINT NOT NULL,
  judge_index INT NOT NULL,
  judge_provider VARCHAR(30),
  judge_model VARCHAR(100),
  total_input_tokens INT DEFAULT 0,
  total_output_tokens INT DEFAULT 0,
  total_llm_calls INT DEFAULT 0,
  total_findings INT DEFAULT 0,
  duration_ms INT DEFAULT 0,
  started_at DATETIME,
  completed_at DATETIME,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY idx_judge_stats_review_judge (review_id, judge_index),
  INDEX idx_judge_stats_review_id (review_id),
  FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE CASCADE
);
-- Document content hash tracking (for deduplication across reviews)
CREATE TABLE IF NOT EXISTS kb_review_doc_hashes (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  agent_id BIGINT NOT NULL,
  source_canonical VARCHAR(500) NOT NULL,
  content_hash VARCHAR(64) NOT NULL,
  review_id BIGINT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_agent_source (agent_id, source_canonical),
  INDEX idx_doc_hashes_agent_id (agent_id),
  FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE
  SET
    NULL
);