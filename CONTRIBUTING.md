# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given. You can contribute in many ways, follow this step-by-step guide to start contributing to Dignea.

---

## Report bugs

Report bugs at https://github.com/amarrerod/digneapy/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it. 

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it. However, you can also propose new features via PR. 
If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Write Documentation

Dignea could always use more documentation and examples, whether as part of the
official digneapy docs, in docstrings, or even on the web in blog posts,
articles, and such.

# Get Started! 

Ready to contribute? Here's how to set up `digneapy` for local development.

1. Fork the `digneapy` repo on GitHub.
2. Clone your fork locally

    $ git clone git@github.com:your_name_here/digneapy.git

3. Install your local copy into a virtualenv. Assuming you have ``uv`` installed, this is how you set up your fork for local development::

    $ cd digneapy 
    $ uv venv
    $ source .venv/bin/activate
    $ uv pip -r requirements.txt
    $ ud pip -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass ruff and the
   tests, including testing other Python versions with tox::

    $ ruff check digneapy tests
    $ uv run pytest --doctest-modules tests 


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for Python 3.12 and 3.13. Check Github Actions and make sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```bash
$ uv run pytest --doctest-modules tests/new_directory/test_new_feature.py
```

