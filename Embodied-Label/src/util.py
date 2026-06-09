from datetime import datetime  
from pathlib import Path
def make_output_dir(base_dir = None, name = "default"):

    if base_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        base_dir = project_root / "logs"
          
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = base_dir / name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

if __name__=="__main__":
    dir=make_output_dir()
    print(dir)