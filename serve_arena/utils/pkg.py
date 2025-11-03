import importlib.util
from importlib.metadata import version
from typing import Optional

# 패키지 버전 정보 확인 방법: https://stackoverflow.com/questions/20180543/how-do-i-check-the-versions-of-python-modules
def find_package_version(pkg: str):
    return version(pkg)

# 패키지 설치 여부 확인 방법: https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed
def find_package(pkg: str, ver: Optional[str] = None, raise_error: bool = False):
    """패키지가 설치되었는지, 버전이 일치하는지 확인하는 함수"""
    is_installed = importlib.util.find_spec(pkg)

    if not is_installed:
        if raise_error:
            raise ImportError(f"{pkg} is not installed")
        else:
            return False
    
    if ver is None:
        return True
    
    is_version_matched = find_package_version(pkg) == ver

    if not is_version_matched:
        if raise_error:
            raise ImportError(f"{pkg} version is not matched")
        else:
            return False
    
    return True

def is_version_at_least(pkg: str, ver: str, raise_error: bool = False):
    is_installed = importlib.util.find_spec(pkg)

    if not is_installed:
        if raise_error:
            raise ImportError(f"{pkg} is not installed")
        else:
            return False
    
    pkg_ver = find_package_version(pkg)
    is_satisfied = pkg_ver >= ver

    if not is_satisfied and raise_error:
        raise RuntimeError(f"{pkg} version {pkg_ver} is lower than required {ver}")
    
    return is_satisfied