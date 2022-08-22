## Team-YYJG

[영양재공 WBS 링크](https://docs.google.com/spreadsheets/d/12GQ9VEX7DxRUBWMm0SNXmG4ZrZy0_ye-cy7Ktq7Y-Qo/edit#gid=785598881)
[Streamlit Share로 배포한 사이트](https://yangchangnaihoby-project-005-try-stre-project-005-in-all-ybqtak.streamlitapp.com)

### 현재까지 구현한 웹 사이트를 본인 컴퓨터에서 실행시키고 싶으신 분들을 위해 설명남깁니다. - 양창은

(08/19 comment) 실행에 앞서 영상 업로드 후 에러 현상은 'Angle only' 옵션으로 전환 후 다시 시도해주세요.

1. 먼저 팀 git repository에 업로드된 최신 버전의 full_program.zip을 다운 받으시고 압축 해제해서 다음과 같이 디렉토리 구조를 배치해주세요.
- 압축 해제하면 디폴트로 이미 배치되어 있어서 상위 디렉토리 경로는 달라도 상관없으며,
- saved_image 폴더는 처음에는 비어있는 폴더입니다.

<img src = 'https://user-images.githubusercontent.com/104478650/185074052-e7f2dd23-63a0-4ec8-9b32-01adffd4a90e.jpg' width = '800'>

2. 그 다음 anaconda에서 본인의 가상환경 상에 pip install streamlit을 통해 streamlit 라이브러리를 설치해주세요.
  - 추가로 구동을 위해 설치해야 할 모듈은 다음과 같습니다. (이미 설치되어 있으면 스킵)
    - Mediapipe
    - Tensorflow
    - PIL
    - OpenCV

3. 아나콘다 명령창(Anaconda Prompt)에서 Project_005_in_all.py 파일이 있는 디렉토리로 이동한 후,
4. 명령창에서 streamlit run Project_005_in_all.py 명령을 입력하시면 자동으로 브라우저에 열리게 됩니다.
5. 업로드된 full_program.zip에는 최신 버전의 코드가 들어있으니 따로 건드리지 않으셔도 됩니다.
6. 실행해보시고 버그 테스트하면서 많은 피드백 부탁드립니다!! 👍👍👍
