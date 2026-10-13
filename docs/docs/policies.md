# Policies

## Using AI for NEMO Cookbook Development

Scientific software developers are increasing using Artificial Inteligence (AI) in the form of Large Language Models (LLMs) to help them in their work. LLMs have received both praise and criticism when it comes to software development.

We accept that NEMO Cookbook contributors will have their own motivation for using or not using AI tools. We have therefore have adopted the same policy as [Parcels](https://docs.parcels-code.org/en/latest/development/policies.html) that we expect all NEMO Cookbook contributors to follow:

> It is ultimately your responsibility to understand the code that you commit.

Please remember that all reviews will be undertaken by human maintainers, so asking us to review code that an LLM generated but you don't understand isn't consistent with our values as an open-source community.

We have also included a [CLAUDE.md](https://github.com/noc-msm/nemo_cookbook/blob/HEAD/CLAUDE.md) file in the repository with additional instructions for AI Agents to follow when contributing to NEMO Cookbook.

## Versioning

NEMO Cookbook follows Calendar Versioning (CalVer), where the version number (e.g., v2026.09.01) is of the form **YYYY.MM.MICRO**.

> **YYYY** - Release year<br>
> **MM** - Release month<br>
> **MICRO** - Release counter within that month, starting at 0<br>

When making breaking code changes to NEMO Cookbook, we will ensure that changes and migration instructions are communicated to users via release notes and migration guides, and informative error messaging / depreceation warning where needed.

If you use NEMO Cookbook in your research and publications, we strongly recommend citing the software including the appropriate version as follows:

> Tooth, O. J., Blaker, A. T., & Coward, A. C. (2026). NEMO Cookbook (Version v2026.09.01) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22688066