rm -f phenonaut*.rst modules.rst
sphinx-apidoc -o . ../src/
rm modules.rst
make html
