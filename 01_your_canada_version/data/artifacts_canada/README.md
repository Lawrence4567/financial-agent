This directory contains the client-case data [客户案例数据] and generated runtime artifacts [运行产物] used by the Canada wealth insights workspace.

## What Is Here

- `user_info.csv`: demo user profile
- `cat.csv`: transaction history
- `account_summary.csv`: account balances and liquidity data
- `portfolio_holdings.csv`: current holdings and exposure data
- `portfolio_performance.csv`: monthly portfolio performance history
- `product_catalog.csv`: representative product catalog
- `reference_rag_index.json`: generated local RAG index
- `audit_logs/`: JSONL audit events written during app runs

## Notes

- No row contains real customer information.
- The records are synthetic [合成数据], but designed to resemble a realistic Canadian household advisory case.
- The product catalog stays institution-neutral [机构中立], so the demo can focus on workflow logic rather than brand-specific offers.
- `reference_rag_index.json` is derived from the files in `../reference_canada/`.
- `audit_logs/` is part of the demo governance layer, not part of the client source data itself.
