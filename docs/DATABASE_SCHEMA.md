# BRN_KB_REVIEW — Database Schema

All tables live in the shared `brn_admin_panel` MySQL database.

---

## 1. Create the DB User

```sql
-- Create a dedicated user for the KB Review service
CREATE USER 'kb_review_svc'@'%' IDENTIFIED BY '<strong-password-here>';

-- Write access to the service tables
GRANT SELECT, INSERT, UPDATE, DELETE ON brn_admin_panel.kb_reviews TO 'kb_review_svc'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON brn_admin_panel.kb_review_issues TO 'kb_review_svc'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON brn_admin_panel.kb_review_judge_results TO 'kb_review_svc'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON brn_admin_panel.kb_review_judge_stats TO 'kb_review_svc'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON brn_admin_panel.kb_review_doc_hashes TO 'kb_review_svc'@'%';

-- ALTER + INDEX on service-owned tables (for auto-migration of new columns and indexes)
GRANT ALTER, INDEX ON brn_admin_panel.kb_reviews TO 'kb_review_svc'@'%';
GRANT ALTER, INDEX ON brn_admin_panel.kb_review_issues TO 'kb_review_svc'@'%';
GRANT ALTER, INDEX ON brn_admin_panel.kb_review_judge_results TO 'kb_review_svc'@'%';
GRANT ALTER, INDEX ON brn_admin_panel.kb_review_judge_stats TO 'kb_review_svc'@'%';
GRANT ALTER, INDEX ON brn_admin_panel.kb_review_doc_hashes TO 'kb_review_svc'@'%';

-- Read access to existing tables
GRANT SELECT ON brn_admin_panel.agents TO 'kb_review_svc'@'%';
GRANT SELECT ON brn_admin_panel.users TO 'kb_review_svc'@'%';
GRANT SELECT ON brn_admin_panel.roles_permissions TO 'kb_review_svc'@'%';
GRANT SELECT ON brn_admin_panel.permissions TO 'kb_review_svc'@'%';

FLUSH PRIVILEGES;
```

---

## 2. Create Tables

Run these in order (foreign key dependencies).

### `kb_reviews`

```sql
CREATE TABLE kb_reviews (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING → RUNNING → COMPLETED | FAILED | PARTIAL
    config_json TEXT,
    total_documents INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    chunks_with_issues INT DEFAULT 0,
    candidate_pairs INT DEFAULT 0,
    url_duplicates_removed INT DEFAULT 0,
    issues_found INT DEFAULT 0,
    issues_resolved INT DEFAULT 0,
    progress INT DEFAULT 0,
    previous_review_id BIGINT,               -- carry-forward: source review
    pairs_reused INT DEFAULT 0,              -- carry-forward: pairs skipped
    pairs_analyzed INT DEFAULT 0,            -- carry-forward: pairs sent to LLM
    docs_changed INT DEFAULT 0,              -- carry-forward: changed documents
    docs_unchanged INT DEFAULT 0,            -- carry-forward: unchanged documents
    created_by_user_id BIGINT,              -- user who started the review (NULL for API key callers)
    error_message TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_kb_reviews_agent_id (agent_id),
    INDEX idx_kb_reviews_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### `kb_review_issues`

```sql
CREATE TABLE kb_review_issues (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    review_id BIGINT NOT NULL,
    issue_type VARCHAR(30) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    doc_a_name VARCHAR(500),
    doc_a_page INT,
    doc_a_excerpt TEXT,
    doc_b_name VARCHAR(500),
    doc_b_page INT,
    doc_b_excerpt TEXT,
    entities_involved TEXT,
    consensus VARCHAR(20),
    judges_flagged INT DEFAULT 0,
    judges_total INT DEFAULT 0,
    carried_forward BOOLEAN DEFAULT FALSE,   -- TRUE if copied from a previous review
    original_review_id BIGINT,               -- chains to the review that first discovered this issue
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',  -- OPEN | RESOLVED | DISMISSED | ACKNOWLEDGED
    -- No DB CHECK constraint — status domain enforced by Pydantic validator at API layer.
    -- MySQL CHECK support varies by version; the validator covers all write paths.
    status_updated_by BIGINT,                -- user_id who changed status (NULL for API key callers)
    status_updated_at DATETIME,              -- when status was last changed
    status_note TEXT,                         -- optional note explaining the status change
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_kb_review_issues_review_id (review_id),
    INDEX idx_kb_review_issues_type (issue_type),
    INDEX idx_kb_review_issues_severity (severity),
    INDEX idx_kb_review_issues_status (status),
    FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### `kb_review_judge_results`

```sql
CREATE TABLE kb_review_judge_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    issue_id BIGINT NOT NULL,
    judge_index INT NOT NULL,
    judge_provider VARCHAR(30),
    judge_model VARCHAR(100),
    detected BOOLEAN NOT NULL DEFAULT FALSE,
    severity VARCHAR(10),
    confidence FLOAT,
    reasoning TEXT,
    input_tokens INT DEFAULT 0,
    output_tokens INT DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_kb_review_judge_results_issue_id (issue_id),
    FOREIGN KEY (issue_id) REFERENCES kb_review_issues(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### `kb_review_judge_stats`

Per-judge aggregate performance stats for each review. One row per judge per review.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGINT PK | Auto-increment |
| `review_id` | BIGINT FK | References `kb_reviews.id` |
| `judge_index` | INT | 0-based index of the judge in the review config |
| `judge_provider` | VARCHAR(30) | Provider name (litellm, fireworks, openrouter) |
| `judge_model` | VARCHAR(100) | Model name |
| `total_input_tokens` | INT | Total input tokens across all LLM calls |
| `total_output_tokens` | INT | Total output tokens across all LLM calls |
| `total_llm_calls` | INT | Number of LLM API calls made |
| `total_findings` | INT | Number of findings produced by this judge |
| `duration_ms` | INT | Wall-clock time in milliseconds |
| `started_at` | DATETIME | When this judge started processing |
| `completed_at` | DATETIME | When this judge finished processing |
| `created_at` | DATETIME | Row creation timestamp |

```sql
CREATE TABLE kb_review_judge_stats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
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
    INDEX idx_judge_stats_review_id (review_id),
    UNIQUE INDEX idx_judge_stats_review_judge (review_id, judge_index),
    FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### `kb_review_doc_hashes`

Per-document content hashes for change detection. One row per (agent, document). Updated in-place when a review completes.

```sql
CREATE TABLE kb_review_doc_hashes (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    agent_id BIGINT NOT NULL,
    source_canonical VARCHAR(500) NOT NULL,   -- canonicalized source name (URL-normalized or filename)
    content_hash VARCHAR(64) NOT NULL,        -- SHA-256 hex of sorted chunk content
    review_id BIGINT,                         -- review that last computed this hash
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME,
    UNIQUE INDEX uq_agent_source (agent_id, source_canonical),
    INDEX idx_doc_hashes_agent_id (agent_id),
    FOREIGN KEY (review_id) REFERENCES kb_reviews(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 3. Alter Existing Tables (if upgrading)

If the tables already exist, run these `ALTER` statements to add the carry-forward columns:

```sql
-- kb_reviews: carry-forward tracking
ALTER TABLE kb_reviews
    ADD COLUMN previous_review_id BIGINT AFTER progress,
    ADD COLUMN pairs_reused INT DEFAULT 0 AFTER previous_review_id,
    ADD COLUMN pairs_analyzed INT DEFAULT 0 AFTER pairs_reused,
    ADD COLUMN docs_changed INT DEFAULT 0 AFTER pairs_analyzed,
    ADD COLUMN docs_unchanged INT DEFAULT 0 AFTER docs_changed;

-- kb_reviews: audit trail
ALTER TABLE kb_reviews
    ADD COLUMN created_by_user_id BIGINT AFTER docs_unchanged;

-- kb_review_issues: carry-forward origin
ALTER TABLE kb_review_issues
    ADD COLUMN carried_forward BOOLEAN DEFAULT FALSE AFTER judges_total,
    ADD COLUMN original_review_id BIGINT AFTER carried_forward;

-- kb_reviews: issue status counts
ALTER TABLE kb_reviews
    ADD COLUMN issues_resolved INT DEFAULT 0 AFTER issues_found;

-- kb_review_issues: status management
ALTER TABLE kb_review_issues
    ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'OPEN' AFTER original_review_id,
    ADD COLUMN status_updated_by BIGINT AFTER status,
    ADD COLUMN status_updated_at DATETIME AFTER status_updated_by,
    ADD COLUMN status_note TEXT AFTER status_updated_at;

-- kb_review_issues: status index (for UI filtering)
CREATE INDEX ix_kb_review_issues_status ON kb_review_issues (status);
```

---

## 4. Rollback

```sql
DROP TABLE IF EXISTS kb_review_doc_hashes;
DROP TABLE IF EXISTS kb_review_judge_stats;
DROP TABLE IF EXISTS kb_review_judge_results;
DROP TABLE IF EXISTS kb_review_issues;
DROP TABLE IF EXISTS kb_reviews;

-- Optional: remove the service user
DROP USER IF EXISTS 'kb_review_svc'@'%';
```

### Partial rollback (remove carry-forward only)

```sql
DROP TABLE IF EXISTS kb_review_doc_hashes;

ALTER TABLE kb_reviews
    DROP COLUMN previous_review_id,
    DROP COLUMN pairs_reused,
    DROP COLUMN pairs_analyzed,
    DROP COLUMN docs_changed,
    DROP COLUMN docs_unchanged;

ALTER TABLE kb_review_issues
    DROP COLUMN carried_forward,
    DROP COLUMN original_review_id;
```

### Partial rollback (remove audit trail only)

```sql
ALTER TABLE kb_reviews DROP COLUMN created_by_user_id;
```

### Partial rollback (remove issue status only)

```sql
ALTER TABLE kb_review_issues
    DROP COLUMN status,
    DROP COLUMN status_updated_by,
    DROP COLUMN status_updated_at,
    DROP COLUMN status_note;

ALTER TABLE kb_reviews DROP COLUMN issues_resolved;
```
