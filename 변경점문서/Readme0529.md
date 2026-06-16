게이지 바추가: AI 판정을 한눈에 파악할 수 있도록 애니메이션 게이지 바 도입.

드래그 앤 드롭(Drag & Drop) 기능: 리포트 창 상단을 잡고 마우스로 화면 내 자유롭게 이동시킬 수 있는 편의 기능 추가.

에러 세분화 추가

리포트 창이 닫힐 때(click, mouseleave 등 모든 케이스) 등록되었던 resize 및 click 이벤트 리스너를 완벽하게 해제하는 cleanupListeners 로직 도입.

좌표 갱신을 위해 무한 반복되던 requestAnimationFrame을 제거하고 이벤트 기반으로 변경.

DOM 렌더링 과부하 방지: MutationObserver가 페이지 전체를 스캔하지 않고, 새로 추가된 노드(addedNodes) 안에서만 미디어를 찾도록 알고리즘 최적화.