import os

def rename_images(folder_path):
    # 변경할 이미지 확장자 지정 (소문자로 작성)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    try:
        # 폴더 내의 모든 파일 가져오기
        files = os.listdir(folder_path)
        
        # 이미지 파일만 필터링 (확장자 확인)
        image_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        # 정렬 (원래 이름 순서대로 번호를 매기기 위함)
        image_files.sort() 
        
        # 파일이 없는 경우 처리
        if not image_files:
            print("해당 폴더에 jpg 또는 png 파일이 없습니다.")
            return

        count = 1
        for filename in image_files:
            # 원본 파일의 확장자만 따로 추출 (예: .jpg, .png)
            ext = os.path.splitext(filename)[1].lower()
            
            # 새 파일 이름 생성 (001, 002, 003... 형식)
            # 무조건 png로 바꾸고 싶다면 ext 대신 '.png'로 적어주시면 됩니다.
            new_name = f"{count:03d}.png"
            # new_name = f"{count:03d}{ext}"
            
            # 기존 파일과 새 파일의 전체 경로 생성
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            
            # 이름 변경 실행
            os.rename(old_file, new_file)
            print(f"✅ 변경 완료: {filename} ➔ {new_name}")
            
            count += 1
            
        print("\n모든 이미지의 이름 변경이 완료되었습니다!")

    except FileNotFoundError:
        print("입력하신 폴더 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# ==========================================
# 실행 부분: 아래 경로를 실제 폴더 경로로 바꿔주세요.
# 윈도우 환경이라면 경로 앞에 'r'을 붙이는 것이 좋습니다.
# ==========================================
target_folder = r"C:\Users\USER\Desktop\Capstone\marathon-path-seg\data\Dataset\masks"  # 예시 경로, 실제 경로로 변경

rename_images(target_folder)