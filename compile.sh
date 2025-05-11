rm -rf `find -type d -name '.ipynb_checkpoints'`
rm -rf `find -type d -name '__pycache__'`
isort -rc -sl .
autoflake --in-place --remove-all-unused-imports=False .
isort -rc -m 3 .
black .