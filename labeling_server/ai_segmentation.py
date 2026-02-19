"""
Step 3: AI Segmentation 추론 모듈
- 학습된 모델로 BBox crop 영역에서 segmentation 수행
- PowerPoint 스타일 +/- 조정 기능 (SAM 스타일 point prompt)
- Flask 웹서버와 연동

사용법:
from ai_segmentation import AISegmentationModel
model = AISegmentationModel(checkpoint_path)
mask = model.predict(cropped_image)
mask = model.refine_with_points(cropped_image, positive_points, negative_points)
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Dict
import json


class AISegmentationModel:
    """AI 기반 Segmentation 모델"""
    
    def __init__(self, 
                 config_path: str = None,
                 checkpoint_path: str = None,
                 device: str = 'cuda:0'):
        """
        Args:
            config_path: MMDetection config 파일 경로
            checkpoint_path: 학습된 checkpoint 경로
            device: cuda:0 또는 cpu
        """
        self.device = device
        self.model = None
        self.config = None
        
        if config_path and checkpoint_path:
            self.load_model(config_path, checkpoint_path)
    
    def load_model(self, config_path: str, checkpoint_path: str):
        """MMDetection 모델 로드"""
        try:
            from mmdet.apis import init_detector
            self.model = init_detector(config_path, checkpoint_path, device=self.device)
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback segmentation (GrabCut)")
            self.model = None
    
    def predict(self, 
                image: np.ndarray, 
                threshold: float = 0.5) -> Dict:
        """
        이미지에서 segmentation 수행
        
        Args:
            image: BGR 이미지 (numpy array)
            threshold: confidence threshold
            
        Returns:
            dict with 'mask', 'polygon', 'confidence'
        """
        if self.model is not None:
            return self._predict_mmdet(image, threshold)
        else:
            return self._predict_fallback(image)
    
    def _predict_mmdet(self, image: np.ndarray, threshold: float) -> Dict:
        """MMDetection 모델로 예측"""
        from mmdet.apis import inference_detector
        
        result = inference_detector(self.model, image)
        
        # 결과 파싱 (instance segmentation)
        pred_instances = result.pred_instances
        
        if len(pred_instances) == 0:
            return self._empty_result(image.shape[:2])
        
        # 가장 confidence 높은 것 선택
        scores = pred_instances.scores.cpu().numpy()
        best_idx = scores.argmax()
        
        if scores[best_idx] < threshold:
            return self._empty_result(image.shape[:2])
        
        mask = pred_instances.masks[best_idx].cpu().numpy()
        polygon = self._mask_to_polygon(mask)
        
        return {
            'mask': mask.astype(np.uint8) * 255,
            'polygon': polygon,
            'confidence': float(scores[best_idx]),
            'category_id': int(pred_instances.labels[best_idx].cpu().numpy())
        }
    
    def _predict_fallback(self, image: np.ndarray) -> Dict:
        """Fallback: GrabCut + Otsu thresholding"""
        h, w = image.shape[:2]
        
        # 1. 이미지 전처리
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. 가장 큰 contour 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_result((h, w))
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 5. 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        polygon = self._mask_to_polygon(mask)
        
        return {
            'mask': mask,
            'polygon': polygon,
            'confidence': 0.7,  # fallback이므로 고정값
            'category_id': 1
        }
    
    def refine_with_points(self,
                           image: np.ndarray,
                           current_mask: np.ndarray,
                           positive_points: List[Tuple[int, int]] = None,
                           negative_points: List[Tuple[int, int]] = None,
                           brush_size: int = 10) -> Dict:
        """
        PowerPoint 스타일 +/- 포인트로 마스크 조정
        
        Args:
            image: BGR 이미지
            current_mask: 현재 마스크 (0-255)
            positive_points: 추가할 영역 포인트 [(x, y), ...]
            negative_points: 제거할 영역 포인트 [(x, y), ...]
            brush_size: 포인트 주변 영역 크기
            
        Returns:
            dict with refined 'mask', 'polygon', etc.
        """
        h, w = image.shape[:2]
        mask = current_mask.copy()
        
        # Positive points: 영역 추가 (GrabCut foreground hint)
        if positive_points:
            for px, py in positive_points:
                # 원형 영역 추가
                cv2.circle(mask, (px, py), brush_size, 255, -1)
                
                # 주변 색상 유사 영역도 추가 (flood fill 스타일)
                self._expand_similar_region(image, mask, px, py, brush_size, add=True)
        
        # Negative points: 영역 제거
        if negative_points:
            for px, py in negative_points:
                cv2.circle(mask, (px, py), brush_size, 0, -1)
                self._expand_similar_region(image, mask, px, py, brush_size, add=False)
        
        # 마스크 정리 (노이즈 제거)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        polygon = self._mask_to_polygon(mask)
        
        return {
            'mask': mask,
            'polygon': polygon,
            'confidence': 0.9,
            'category_id': 1
        }
    
    def _expand_similar_region(self, 
                                image: np.ndarray,
                                mask: np.ndarray,
                                cx: int, cy: int,
                                radius: int,
                                add: bool = True,
                                color_threshold: int = 30):
        """색상 유사 영역 확장/축소 (간단한 region growing)"""
        h, w = image.shape[:2]
        
        # 중심점 색상
        if 0 <= cx < w and 0 <= cy < h:
            center_color = image[cy, cx].astype(np.float32)
        else:
            return
        
        # 주변 영역 검사
        search_radius = radius * 3
        y_start = max(0, cy - search_radius)
        y_end = min(h, cy + search_radius)
        x_start = max(0, cx - search_radius)
        x_end = min(w, cx + search_radius)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 거리 체크
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist > search_radius:
                    continue
                
                # 색상 유사도 체크
                pixel_color = image[y, x].astype(np.float32)
                color_diff = np.sqrt(np.sum((pixel_color - center_color) ** 2))
                
                if color_diff < color_threshold:
                    if add:
                        mask[y, x] = 255
                    else:
                        mask[y, x] = 0
    
    def _mask_to_polygon(self, mask: np.ndarray, simplify: bool = True) -> List[float]:
        """마스크를 COCO polygon 형식으로 변환"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # 가장 큰 contour 선택
        largest = max(contours, key=cv2.contourArea)
        
        if simplify:
            # Douglas-Peucker simplification
            epsilon = 0.005 * cv2.arcLength(largest, True)
            largest = cv2.approxPolyDP(largest, epsilon, True)
        
        # COCO format: [x1, y1, x2, y2, ...]
        polygon = largest.flatten().tolist()
        
        return polygon
    
    def _empty_result(self, shape: Tuple[int, int]) -> Dict:
        """빈 결과 반환"""
        return {
            'mask': np.zeros(shape, dtype=np.uint8),
            'polygon': [],
            'confidence': 0.0,
            'category_id': 0
        }


class SegmentationRefiner:
    """
    Interactive segmentation refinement
    PowerPoint 배경제거 스타일의 +/- 조정
    """
    
    def __init__(self):
        self.current_mask = None
        self.history = []  # Undo용
        self.positive_points = []
        self.negative_points = []
    
    def set_initial_mask(self, mask: np.ndarray):
        """초기 마스크 설정"""
        self.current_mask = mask.copy()
        self.history = [mask.copy()]
        self.positive_points = []
        self.negative_points = []
    
    def add_positive_point(self, x: int, y: int):
        """+ 포인트 추가 (전경 힌트)"""
        self.positive_points.append((x, y))
    
    def add_negative_point(self, x: int, y: int):
        """- 포인트 추가 (배경 힌트)"""
        self.negative_points.append((x, y))
    
    def refine(self, image: np.ndarray, model: AISegmentationModel) -> np.ndarray:
        """현재 포인트들로 마스크 업데이트"""
        if self.current_mask is None:
            return None
        
        result = model.refine_with_points(
            image,
            self.current_mask,
            self.positive_points,
            self.negative_points
        )
        
        # History에 저장
        self.history.append(self.current_mask.copy())
        self.current_mask = result['mask']
        
        # 포인트 초기화
        self.positive_points = []
        self.negative_points = []
        
        return result
    
    def undo(self) -> Optional[np.ndarray]:
        """이전 상태로 되돌리기"""
        if len(self.history) > 1:
            self.history.pop()
            self.current_mask = self.history[-1].copy()
            return self.current_mask
        return None
    
    def get_polygon(self) -> List[float]:
        """현재 마스크의 polygon 반환"""
        if self.current_mask is None:
            return []
        
        contours, _ = cv2.findContours(
            self.current_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        return simplified.flatten().tolist()


# Flask API endpoint 예시
def create_segmentation_api():
    """Flask API 생성 예시"""
    from flask import Blueprint, request, jsonify
    import base64
    
    bp = Blueprint('segmentation', __name__)
    
    # 글로벌 모델 (lazy loading)
    _model = None
    
    def get_model():
        nonlocal _model
        if _model is None:
            _model = AISegmentationModel()
            # 모델 로드 시도 (checkpoint 있으면)
            try:
                _model.load_model(
                    '/data2/project/2026winter/jjh0709/mmdetection/configs/vision/crop_seg_cable.py',
                    '/data2/project/2026winter/jjh0709/mmdetection/work_dirs/crop_seg_cable_r50/best_segm_mAP_epoch_XX.pth'
                )
            except:
                print("Using fallback segmentation")
        return _model
    
    @bp.route('/segment', methods=['POST'])
    def segment():
        """
        BBox crop 이미지에서 segmentation 수행
        
        Request:
            - image: base64 encoded image
            - threshold: confidence threshold (default 0.5)
        
        Response:
            - polygon: [x1, y1, x2, y2, ...]
            - confidence: float
            - mask_base64: base64 encoded mask image
        """
        data = request.json
        
        # 이미지 디코딩
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        threshold = data.get('threshold', 0.5)
        
        # 추론
        model = get_model()
        result = model.predict(image, threshold)
        
        # 마스크 인코딩
        _, mask_encoded = cv2.imencode('.png', result['mask'])
        mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')
        
        return jsonify({
            'polygon': result['polygon'],
            'confidence': result['confidence'],
            'mask_base64': mask_base64,
            'category_id': result['category_id']
        })
    
    @bp.route('/refine', methods=['POST'])
    def refine():
        """
        +/- 포인트로 마스크 조정
        
        Request:
            - image: base64 encoded image
            - current_mask: base64 encoded current mask
            - positive_points: [[x, y], ...]
            - negative_points: [[x, y], ...]
            - brush_size: int
        
        Response:
            - polygon: refined polygon
            - mask_base64: refined mask
        """
        data = request.json
        
        # 이미지 디코딩
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 마스크 디코딩
        mask_data = base64.b64decode(data['current_mask'])
        mask_nparr = np.frombuffer(mask_data, np.uint8)
        current_mask = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
        
        positive_points = [tuple(p) for p in data.get('positive_points', [])]
        negative_points = [tuple(p) for p in data.get('negative_points', [])]
        brush_size = data.get('brush_size', 10)
        
        # 조정
        model = get_model()
        result = model.refine_with_points(
            image, current_mask,
            positive_points, negative_points,
            brush_size
        )
        
        # 마스크 인코딩
        _, mask_encoded = cv2.imencode('.png', result['mask'])
        mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')
        
        return jsonify({
            'polygon': result['polygon'],
            'mask_base64': mask_base64
        })
    
    return bp


if __name__ == '__main__':
    # 테스트
    print("Testing AISegmentationModel...")
    
    # 더미 이미지로 테스트
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    model = AISegmentationModel()  # fallback 모드
    result = model.predict(test_img)
    
    print(f"Mask shape: {result['mask'].shape}")
    print(f"Polygon points: {len(result['polygon']) // 2}")
    print(f"Confidence: {result['confidence']}")
    
    print("\nTesting refinement...")
    refiner = SegmentationRefiner()
    refiner.set_initial_mask(result['mask'])
    refiner.add_positive_point(128, 128)
    refined = refiner.refine(test_img, model)
    print(f"Refined mask shape: {refined['mask'].shape}")
