## Run sample

```bash
# View file
python src/engraver.py --input samples/six.svg --scale 10000 --tol 0.1

# Export without viewing
python src/engraver.py --input samples/six.svg --scale 10000 --tol 0.1 --export-json output/six-toolpaths.json --no-view
```

## Pytest

### How to use
1. Put your implementation in a module importable as `shapefit`
   (or edit `tests/conftest.py` to point at your module name).
   
2. Run tests:
   ```bash
   pip install -e .
   pip install pytest
   pytest