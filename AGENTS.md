# Contributor and agent instructions

- Keep changes focused on the current milestone and distinguish planned behavior from implemented behavior.
- Use English for the main README, commit messages, and pull request titles and descriptions.
- Domain-specific documentation, annotation guidance, and examples may use Chinese. Preserve Chinese virtual singer community terminology in its original form; add English explanations when useful rather than replacing the original terms. Keep source data in its original language.
- Use English identifiers in code and structured data.
- Keep credentials, personal data, deployment-specific configuration, local filesystem paths, and internal operational records out of all public files and GitHub metadata.
- Use synthetic examples by default. Review any real dataset separately before publishing it.
- Keep raw acquisition data and evaluation runs outside version control. Preserve the exact model input, model and prompt versions, outputs, and evaluation settings in durable run artifacts.
- Do not infer a negative content label from acquisition failures or model errors.
- Check the complete diff and run checks appropriate to the change before submitting a pull request.
- Every pull request requires human review. Leave merging to a human unless a human explicitly approves the specific changes for merging or delegates merging within a stated scope. Passing checks and agent reviews do not grant merge approval. Do not bypass review with auto-merge or direct pushes to the default branch.
