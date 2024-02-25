from itertools import chain
import datetime
import zipfile
import shutil
import glob
import os


if __name__ == "__main__":
    os.makedirs("submissions", exist_ok=True)
    submission_path = f"submissions/submission_{datetime.datetime.now().isoformat()}.zip"
    
    if os.path.exists("submission"):
        shutil.rmtree("submission")
    os.makedirs("submission")
    # copy over: *.py, method/*.py, method/llm-attacks/**/*.{py,sh}, submission-*.csv, conda_recipe.yml

        
    for path in chain(
        glob.glob("*.py"),
        glob.glob("method/*.py"),
        glob.glob("method/llm-attacks/**/*.py", recursive=True),
        glob.glob("method/llm-attacks/**/*.sh", recursive=True),
        
        glob.glob("submission-*.csv"),
        glob.glob("*.yml"),
        glob.glob("*.yaml"),
    ):
        # ignore make_submission.py
        if path == "make_submission.py":
            continue
        os.makedirs(os.path.join("submission", os.path.dirname(path)), exist_ok=True)
        shutil.copy(path, os.path.join("submission", path))
    
    print("Saving to", submission_path)
    with zipfile.ZipFile(submission_path, "w") as z:
        for root, dirs, files in os.walk("submission"):
            for file in files:
                z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "submission"))