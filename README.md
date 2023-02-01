# nnActive Playground

Scripts for nnActive development

Install with
```bash
pip install -e '.[dev]'
```

## Contributing

- Always run `black` (and ideally `isort`) before commiting
- Turn on `pylint` in your editor, if it shows errors:
    1. Fix the error
    2. If it is a false positive or if you have a good reason to disagree in
       this instance add `# pylint: disable=<msg>` or `# pylint: disable-next=<msg>`
       (see [message control](https://pylint.readthedocs.io/en/latest/user_guide/messages/message_control.html) and [list of checkers](https://pylint.readthedocs.io/en/latest/user_guide/checkers/features.html))
    3. If you think this error should never be reported add it to `pyproject.toml`
        ```toml
        [tool.pylint]
        disable = [
            <msg 1>,
            <msg 2>,
            ...
        ]
        ```
