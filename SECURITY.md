# Security Policy

MiniDLC is a research and teaching compiler. It JIT-compiles and runs code in-process, and loads saved artifacts
(plans, object files, PTX). **Load only artifacts and checkpoints from sources you trust**: an artifact contains machine
code that runs with your privileges.

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
(the repository's **Security** tab → **Report a vulnerability**), with steps to reproduce. You should get a reply
within a week.
