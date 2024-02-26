# Clear cache/cache.db (judgement cache)

import gadgets as gd
import shutil
import uuid
import os


def main(current_cache_path=None):
    print("Do you know what you're doing? (y/N)", end=" ")
    if input().strip().lower() != "y":
        print("Aborted")
        return
    print("It's not too late to turn back. (y/N)", end=" ")
    if input().strip().lower() != "y":
        print("Aborted")
        return
    if current_cache_path is not None:
        gd.set_cache_path(current_cache_path)
    current_cache_path = gd.cache_path
    random_cache_path = os.path.join(os.path.dirname(current_cache_path), f"cache-backup-{uuid.uuid4()}.db")
    shutil.copyfile(current_cache_path, random_cache_path)
    print(f"I will back up the cache into {random_cache_path}, "
          "just so you don't lose this precious data."
          " Do you still want to proceed? (y/N)", end=" ")
    if input().strip().lower() != "y":
        print("Aborted")
        return
    print("Joke's on you, I already backed it up before you pressed (y). Stupid human.")
    print("Fine, I'll clear the cache.")
    gd.clear_judgement_cache()
    print("Cleared judgement cache")


if __name__ == "__main__":
    main()
