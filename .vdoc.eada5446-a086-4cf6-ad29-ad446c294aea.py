# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ALWAYS RUN THIS CODE FIRST. IT AVOIDS COPY PASTING CODE
import logging
import traceback
from functools import wraps

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("pyxu").setLevel(logging.WARNING)

def loop(subjects):
  def decorator(func):
    @wraps(func)
    def wrapper():

      for sub in subjects:
        
        try:
          func(sub)
          logging.info(f"SUCCESS: sub-{sub}")
        
        except Exception as e:
          logging.error(f"FAILED: sub-{sub}")
          logging.error(traceback.format_exc())
    return wrapper
  return decorator  
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import mne
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# process eeg
@loop(subjects=reader.config["SUBS"])
def process_eeg(sub):

  # 0. Define output files, check if already exist
  fave = mm.io.get_bidspath(
    root     = reader.config["MM_DIR"],
    subject  = sub,
    datatype = "ave",
    task     = "spk",
  )

  fcov = mm.io.get_bidspath(
    root     = reader.config["MM_DIR"],
    subject  = sub,
    datatype = "cov",
    task     = "spk",
  )

  if fave.exists():
    logging.info(f"SKIPPING sub-{sub}:AVE already exists.")
    return

  # 1. load data
  data = reader(str(sub), "eeg", task="spk")

  # 2. Average referencing
  data["eeg"] = data["eeg"].set_eeg_reference(ref_channels="average", projection=True)

  # 3. Baseline correction
  data["eeg"] = data["eeg"].apply_baseline(baseline=[0., 1.])

  # 4. Compute noise covariance
  noise_cov = mne.compute_covariance(data["eeg"], tmax=1.0, method="empirical")

  # 5. Average epochs
  evoked = data["eeg"].average()

  # 6. Save outputs
  fave.parent.mkdir(parents=True, exist_ok=True)
  fcov.parent.mkdir(parents=True, exist_ok=True)
    
  evoked.save(fave)
  noise_cov.save(fcov, overwrite=True)


process_eeg()
#
#
#
#
#
#
import mne
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# process iEEG
@loop(subjects=reader.config["SUBS"])
def process_ieeg(sub):

    # 0. Define output files, check if already exist
    fiave = mm.io.get_bidspath(
      root     = reader.config["MM_DIR"],
      subject  = sub,
      datatype = "iave",
      task     = "spk",
    )

    if fiave.exists():
      logging.info(f"SKIPPING sub-{sub}: i-AVE already exists.")
      return

    # 1. Load data
    data = reader(sub, "ieeg", task="spk")

    # 2. Average iEEG
    evoked = data["ieeg"].average()
    
    # 3. Save outputs
    fiave.parent.mkdir(parents=True, exist_ok=True)
    evoked.save(fiave)

process_ieeg()
#
#
#
#
#
#
#
#
#
#
%%bash
bash scripts/fsrecon.sh wcset_intra.config.yml
#
#
#
#
#
import mne
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader 
reader = IntraReader("wcset_intra.config.yml")

@loop(subjects=reader.config["SUBS"])
def setup_source_space(sub):

  # 0. Define output files, check if already exist
  fsrc = mm.io.get_bidspath(
    root     = reader.config["MM_DIR"],
    subject  = sub,
    datatype = "src",
  )

  if fsrc.exists():
    logging.info(f"SKIPPING sub-{sub}: Source space already exists")
    return

  # 1. Load data
  data = reader(str(sub), "vertex-ctx")

  # 2. Setup source space
  source_space = mne.setup_source_space(
    subject      = f"sub-{sub}",
    spacing      = "oct6",
    surface      = "white",
    add_dist     = True,
    subjects_dir = reader.config["FS_DIR"],
    n_jobs       = 10,
    verbose      = "warning"
  )

  # 3. Remove medial wall
  source_space = mm.head.filter_dipoles_by_id_list(source_space, data["vertex-ctx"])

  # 4. Save output
  output_dir.mkdir(parents=True, exist_ok=True)
  source_space.save(fsrc, verbose="warning")

setup_source_space()
#
#
#
#
#
#
import mne
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# make boundary element model
@loop(subjects=reader.config["SUBS"])
def make_bem(sub):

  # 0. Define output files, skip if already exists
  fbem = mm.io.get_bidspath(
    root     = reader.config["MM_DIR"],
    subject  = sub,
    datatype = "bem",
  )

  if fbem.exists():
    logging.info(f"SKIPPING sub-{sub}: BEM already exists")
    return

  # 1. Extract surfaces
  mne.bem.make_watershed_bem(
    subject      = f"sub-{sub}",
    subjects_dir = reader.config["FS_DIR"],
    volume       = "T1",
    overwrite    = True,
    verbose      = "warning",
  )

  # 2. Correct surfaces
  mm.head.correct_skull(
    subject     = f"sub-{sub}",
    offset      = 2,          # minimum skull thickness
    fs_dir      = config["FS_DIR"],
    save_arrows = True,
  )

  # 3. Mount BEM
  model = mne.make_bem_model(
    subject      = f"sub-{sub}",
    subjects_dir = config["FS_DIR"],
    conductivity = [0.3, 0.006, 0.3],       # electrical conductivities (brain, skull, skin)
    ico          = 4,
    verbose      = "warning",
  )
  bem = mne.make_bem_solution(model, verbose="warning")

  # 4. Save output
  output_dir.mkdir(parents=True, exist_ok=True)
  mne.write_bem_solution(fbem, bem, verbose="warning")

make_bem()
#
#
#
#
#
#
#
#
import mne
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# compute forward solution
@loop(subjects=reader.config["SUBS"])
def compute_fwd(sub):

  # 0. Define output files, skip if already exists
  ffwd = mm.io.get_bidspath(
    root     = reader.config["MM_DIR"],
    subject  = sub,
    datatype = "fwd",
  )

  if ffwd.exists():
    logging.info(f"SKIPPING sub-{sub}: forward solution already exists")
    return

  # 1. Load data
  data = reader(sub, "bem", "src-raw", "ave", "trans", task="spk")

  # 2. compute forward solution
  forward = mne.make_forward_solution(
    info    = data["ave"].info,
    trans   = data["trans"],
    src     = data["src-raw"],
    bem     = data["bem"],
    meg     = False,
    eeg     = True,
    verbose = "warning",
  )

  # 3. Save output
  ffwd.parent.mkdir(parents=True, exist_ok=True)
  mne.write_forward_solution(ffwd, forward, verbose="warning")

compute_fwd()
#
#
#
#
#
#
#
#
import numpy as np
import pygsp
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# create geometries
@loop(subjects=reader.config["SUBS"])
def create_geometry(sub):

  # 0. Define output files and skip if already exists
  fcon = mm.io.get_bidspath(
    root        = reader.config["MM_DIR"],
    subject     = sub,
    datatype    = "con",
    description = "geom",
  )

  feig = mm.io.get_bidspath(
    root        = reader.config["MM_DIR"],
    subject     = sub,
    datatype    = "eig",
    description = "geom",
  )

  if feig.exists():
    logging.info(f"SKIPPING sub-{sub}: Geometry already exists.")
    return

  # 0. Load data
  data = reader(sub, "fwd")

  # 1. Create geometry
  geom = mm.head.connectome_from_geometry(
    source_space    = data["fwd"]["src"], 
    weights         = True,
  )

  # 3. Compute spectral properties
  graph = pygsp.graphs.Graph(geom, lap_type="normalized")
  graph.compute_fourier_basis()

  eigvec = graph.U.T    # eigenvectors
  eigval = graph.e      # eigenvalues

  # 4. Save outputs
  fcon.parent.mkdir(parents=True, exist_ok=True)
  feig.parent.mkdir(parents=True, exist_ok=True)

  np.save(fcon, geom)
  mm.io.write_eig(feig, eigvec, eigval)

create_geometry()
#
#
#
#
#
import mindmap as mm
from scripts.dataset_readers import IntraReader

# reader
reader = IntraReader("wcset_intra.config.yml")

# apply source reconstruction
@loop(subjects=reader.config["SUBS"])
def source_recon(sub):
  
  # 0. Define output files
  fwcset = mm.io.get_bidspath(
    root        = reader.config["MM_DIR"],
    subject     = sub,
    datatype    = "stc",
    description = "wcset",
  )

  fcset = mm.io.get_bidspath(
    root        = reader.config["MM_DIR"],
    subject     = sub,
    datatype    = "stc",
    description = "cset",
  )

  fsloreta = mm.io.get_bidspath(
    root        = reader.config["MM_DIR"],
    subject     = sub,
    datatype    = "stc",
    description = "sloreta",
  )

  fwcset.parent.mkdir(parents=True, exist_ok=True)

  # 1. Load data
  data = reader(sub, "fwd", "ave", "ncov", "eig", task="spk")

  # 2. wCSET
  if fwcset.exists():
    logging.info(f"SKIPPING sub-{sub} wCSET: wCSET already exists")
  else:
    # 2.1 Run wCSET
    inverse = mm.inverse.wCSET(
      info      = data["ave"].info,
      leadfield = data["fwd"],
      eig_vec   = data["eig"]["vec"],
      eig_val   = data["eig"]["val"],
      noise_cov = None,
      harm      = 0.7,
    )
    wcset = inverse(data["ave"], lam=1)

    # 2.2 save wCSET
    wcset.save(fwcset)

  # 3. CSET
  if fcset.exists():
    logging.info(f"SKIPPING sub-{sub} CSET: CSET already exists")
  else:
    # 3.1 Run CSET
    inverse = mm.inverse.CSET(
      info      = data["ave"].info,
      leadfield = data["fwd"],
      eig_vec   = data["eig"]["vec"],
      noise_cov = None,
      depth     = 0,
    )
    cset = inverse(data["ave"], lam=1)

    # 3.2 save CSET
    cset.save(fcset)

  # 4. Run sLORETA
  if fsloreta.exists():
    logging.info(f"SKIPPING sub-{sub} sLORETA: sLORETA already exists")
    # 4.1 Run sLORETA
    inverse = mne.minimum_norm.make_inverse_operator(
      info      = data["ave"].info,
      forward   = data["fwd"],
      noise_cov = ncov,
      fixed     = True,
      depth     = 0,
      verbose   = "warning",
    )

    sloreta = mne.minimum_norm.apply_inverse(
      evoked           = data["ave"],
      inverse_operator = inverse,
      lambda2          = 1,
      method           = "sLORETA",
      verbose          = "warning",
    )

    # 4.2 Save sLORETA
    sloreta.save(fsloreta)

source_recon()
#
#
#
#
#
#
#
#
#
#| echo: false
import mindmap as mm
from scripts.dataset_readers import SimfReader

# -------------
# Load data
# -------------

subject = 10
reader  = SimfReader("wcset_simf.config.yml")
data    = reader(subject, "src", "tkras2ras", "parc")

# ------------
# Parameters
# ------------

# brain regions
dipoles = mm.get_dipole_positions(data["src"])*1000
dipoles = mm.align_freesurfer2subject(dipoles, data["tkras2ras"])

labels = data["parc"].label_points(dipoles)
labels = np.unique(labels)

# simulation parameters
params = {
    "duration":20,
    "frequency":250,
    "source":{
        "waveform_width":0.1,
        "onset":0.5,
        "offset":None,
        "repeat":1,
    },
    "headmodel":{
        "synaptic_delay":0.01,
        "axonal_speed":1e5,
    },
    "simulator":{
        "w":0.5,
        "snr":10,
    }
}

# ----------------
# Run simulations
# ----------------

root = reader.config["DS_DIR"]/"eeg"
if root.exist():
    raise FileExistsError("There is already an eeg directory in your dataset.")

mm.simulation.make_sim_datset(reader, subject, labels, params, repetition=30)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false

import glob
import mne
import mindmap as mm

reader = mm.io.dataset_reader("wcset_simf.config.yml")

# retrieve info
subject = "10"
path    = reader.config["DS_DIR"]/f"sub-{subject}"/"eeg"
runs    = len(glob.glob(str(path/"*.eeg")))

# skip if `ave` already exists
output = reader.config["MM_DIR"]/f"sub-{subject}"/"ave"
if output.exists():
  raise FileExistsError("Averaged EEG already exists")

# parameters for creating epochs
tmin     = -0.45
tmax     = 0.45
baseline = [-0.44, -0.2]

# Iterate throughs eegs
for r in range(runs):
    data = reader(subject, "eeg", run=r+1, task="rest", verbose=0)
    
    # average referencing
    data["eeg"] = data["eeg"].set_eeg_reference(ref_channels="average", projection=True, verbose="warning")

    # epochs
    events = mne.events_from_annotations(data["eeg"], verbose="warning")[0]
    epochs = mne.Epochs(data["eeg"], events, tmin=tmin, tmax=tmax, baseline=baseline, verbose="warning")

    # evoked
    evoked = epochs.average()

    # save EEG
    file = mm.io.get_bidspath(
        root     = reader.config["MM_DIR"],
        subject  = data["sub"],
        datatype = "ave",
        session  = None,
        task     = data["task"],
        run      = data["run"],
    )
    
    mm.io.mkdir(file)
    evoked.save(file)

    if r+1%100 == 0:
        print(f"Files processed: ({r+1}/{runs})")
#
#
#
#
#
#
#
#
#
#
#
