"""Build an isolated read-only extension against the installed SAPIEN headers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig
import sapien
import torch

root = Path(__file__).resolve().parent
package = Path(sapien.__file__).parent
libs = package.parent / 'sapien.libs'
source = root / 'native_joint_readback.cpp'
output = root / ('af_native_joint_readback' + sysconfig.get_config_var('EXT_SUFFIX'))
command = ['g++', '-O2', '-shared', '-std=c++17', '-fPIC', '-DNDEBUG', '-fabi-version=14',
           '-I' + sysconfig.get_path('include'),
           '-I' + str(Path(torch.__file__).parent / 'include'),
           '-I' + str(package / 'include'),
           '-I' + str(package / 'include/physx/include'),
           '-I' + str(root / 'eigen_3_4_0'), str(source),
           '-L' + str(libs), '-lsapien', '-Wl,-rpath,' + str(libs), '-o', str(output)]
print('BUILD_COMMAND ' + json.dumps(command), flush=True)
subprocess.run(command, check=True)
(root / 'NATIVE_BUILD.json').write_text(json.dumps({'command': command,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'sapien_version': sapien.__version__}, indent=2))
import af_native_joint_readback
print('NATIVE_IMPORT_PASSED', af_native_joint_readback.__file__)
