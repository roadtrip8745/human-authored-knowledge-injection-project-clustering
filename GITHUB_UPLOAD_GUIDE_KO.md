# GitHub 업로드 가이드

## 1. 업로드 전 확인

이 폴더는 GitHub 공개용으로 정리한 재현성 패키지입니다.

공개 포함:

- 합성 메일 데이터
- 합성 메일에서 추출된 mention-level 데이터
- human-authored knowledge 파일
- deterministic clustering/evaluation 코드
- IEEE Access 실험 결과 CSV/JSON/MD
- Outlook 원본이 아닌 집계 distribution bin

공개 제외:

- 실제 사내 메일 원문
- Outlook 수집기 전체 코드
- LLM API 호출 코드와 API key
- 사내 서비스 설정
- 로컬 절대경로, 로그, 캐시
- 이전 JIPS baseline 전체 소스

업로드 직전에 아래 스캔을 한 번 더 실행하세요.

```powershell
cd <path-to-public-package>
rg -n --glob "!GITHUB_UPLOAD_GUIDE_KO.md" "gsk_|api\.groq|GROQ|API_KEY|OPENAI_API_KEY|MTR_LLM_API_KEY|sk-|credential|D:\\|C:\\|Users\\" .
```

아무 출력도 없어야 안전합니다.

## 2. GitHub 저장소 만들기

GitHub에서 새 repository를 만듭니다.

권장 이름:

`human-authored-knowledge-injection-project-clustering`

처음에는 `Private`로 만들고, 한 번 더 검토한 뒤 `Public`으로 전환하는 것을 권장합니다.

## 3. 로컬에서 push

```powershell
cd <path-to-public-package>
git init
git add .
git status
git commit -m "Release reproducibility artifacts for human-authored knowledge injection"
git branch -M main
git remote add origin https://github.com/<YOUR_ID>/human-authored-knowledge-injection-project-clustering.git
git push -u origin main
```

`<YOUR_ID>`는 본인 GitHub 계정명으로 바꾸면 됩니다.

## 4. DOI가 필요할 때

IEEE Access 논문에서 재현성 링크를 더 탄탄하게 보이게 하려면 GitHub만 두는 것보다 Zenodo DOI까지 붙이는 편이 좋습니다.

절차:

1. GitHub repository를 만든다.
2. Zenodo에서 GitHub 연동을 켠다.
3. GitHub에서 `v1.0.0` release를 만든다.
4. Zenodo가 생성한 DOI를 논문 `Data and Code Availability`에 적는다.

논문 문구 예시:

```tex
The reproducibility package containing the synthetic corpus, mention-level records,
human-authored knowledge files, deterministic evaluation code, and result tables is
available at: https://github.com/<YOUR_ID>/<REPO_NAME>. A DOI-archived snapshot
is available at: https://doi.org/<DOI>.
```

## 5. 라이선스

공개 전 라이선스를 정해야 합니다.

무난한 선택:

- 코드: MIT License 또는 Apache License 2.0
- 합성 데이터/결과표: CC BY 4.0

라이선스를 정하지 않고 공개하면 다른 연구자가 재사용할 권한이 불명확해져 재현성 측면에서 아쉽게 보일 수 있습니다.
