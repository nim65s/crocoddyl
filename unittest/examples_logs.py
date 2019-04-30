import os
import sys
from difflib import ndiff
from subprocess import PIPE, Popen

example, project = os.environ['CROCODDYL_EXAMPLE'], os.environ['PROJECT_SOURCE_DIR']

process = Popen([sys.executable, os.path.join(project, 'examples', example + '.py')], stdout=PIPE, stderr=sys.stderr)
output, _ = process.communicate()

if process.returncode != 0:
    print('Example "%s" failed, with the following output:' % example)
    print(output)
    sys.exit(process.returncode)

print('Example "%s" succeeded. Checking logs...' % example)

with open(os.path.join(project, 'examples/log', example + '.log')) as f:
    expected = f.read()

diff = list(ndiff(expected.splitlines(), output.splitlines()))
print('\n'.join(diff))
sys.exit(len(list(line for line in diff if not line.startswith(' '))))
