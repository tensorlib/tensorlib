make html
bash fixup_docs.sh

cd _build/html
cp -pr * /path/to/your/tensorlib.github.io fork
cd /path/to/your/tensorlib.github.io fork
git add *
git commit -a
git push origin master

Make a PR. Merge PR. Celebrate :)
