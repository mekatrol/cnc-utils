## Run sample

```bash
# View file
python engraver.py samples/six.svg --scale 10000 --tol 0.1

# Export without viewing
python engraver.py samples/six.svg --scale 10000 --tol 0.1 --export-json output/six-toolpaths.json --no-view
```