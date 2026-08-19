"""
Helper to attempt installing optional binary wheels (xformers, pytorch3d) without triggering source builds.
Run this in Colab to attempt binary-only installs and print the results.
"""
import subprocess
import sys

WHEELS = [
    {
        'name': 'xformers',
        'spec': 'xformers',
        'find_links': 'https://download.pytorch.org/whl/torch_stable.html',
    },
    {
        'name': 'pytorch3d',
        'spec': 'pytorch3d',
        'find_links': 'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3-none-any/',
    },
]

def try_install(spec, find_links=None):
    cmd = [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', spec]
    if find_links:
        cmd += ['-f', find_links]
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    ok = res.returncode == 0
    if ok:
        print(f'Installed {spec} successfully')
    else:
        print(f'Failed to install {spec} (binary only). Output:')
        print(res.stdout)
        print(res.stderr)
    return ok

if __name__ == '__main__':
    for w in WHEELS:
        try_install(w['spec'], w.get('find_links'))
    print('Done. If installs failed, proceed without xformers/pytorch3d — the code will still run.')
