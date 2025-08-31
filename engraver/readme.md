## Run sample

```bash
# View file
python src/engraver.py --input samples/six.svg --scale 10000 --tol 0.1

# Export without viewing
python src/engraver.py --input samples/six.svg --scale 10000 --tol 0.1 --export-json output/six-toolpaths.json --no-view
```