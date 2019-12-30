#!/bin/sh
# Copy the Sphinx API doc

dest=../../api
rm -rf $dest
cp -r api/_build/html $dest

cat > $dest/README <<EOF
This directory contains the API documentation for the package.
The documentation is automatically generated by ../src/api/make.py.
EOF
git add $dest
git commit -am 'Added new official API doc files.'