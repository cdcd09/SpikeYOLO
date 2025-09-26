#!/usr/bin/env python3
"""
Roboflow 데이터셋 다운로드 스크립트
"""
import os
import sys

# roboflow 패키지가 없으면 설치
try:
    from roboflow import Roboflow
except ImportError:
    print("roboflow 패키지가 설치되지 않았습니다. 설치 중...")
    os.system("pip install roboflow")
    from roboflow import Roboflow

def download_dataset():
    # API 키와 프로젝트 정보
    api_key = "HwIYu3zVL2SPcVB9c21X"
    workspace_name = "competition-o98lu"
    project_name = "obstacles-yywad"
    version_number = 5
    format_type = "yolov5"
    
    # 다운로드 경로를 /data로 설정
    download_path = "/data"
    
    # 경로가 존재하는지 확인
    if not os.path.exists(download_path):
        print(f"경로 {download_path}가 존재하지 않습니다.")
        return False
    
    try:
        print("Roboflow 클라이언트 초기화 중...")
        rf = Roboflow(api_key=api_key)
        
        print(f"프로젝트 {workspace_name}/{project_name} 접근 중...")
        project = rf.workspace(workspace_name).project(project_name)
        
        print(f"버전 {version_number} 로드 중...")
        version = project.version(version_number)
        
        # 현재 디렉토리를 /data로 변경
        os.chdir(download_path)
        
        print(f"데이터셋 다운로드 시작... (형식: {format_type})")
        print(f"다운로드 위치: {download_path}")
        
        dataset = version.download(format_type)
        
        print("데이터셋 다운로드 완료!")
        print(f"데이터셋 위치: {dataset.location}")
        
        # 다운로드된 파일들 확인
        print("\n다운로드된 내용:")
        for root, dirs, files in os.walk(dataset.location):
            level = root.replace(dataset.location, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("데이터셋 다운로드가 성공적으로 완료되었습니다!")
    else:
        print("데이터셋 다운로드에 실패했습니다.")
        sys.exit(1)