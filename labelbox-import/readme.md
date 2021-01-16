# Import Planes to Labelbox

Copy **labelbox.env.example** to **labelbox.env**

Add your Labelbox API Key. You will need a Pro or Enterprise account to do the (Model Assisted Labeling)[https://labelbox.com/docs/automation/model-assisted-labeling]

````bash
source labelbox.env
```

```bash
python3 create.py
```

```bash
python3 import.py --filePath=/path/to/files
```
