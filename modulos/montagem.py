import cv2
import numpy as np
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import (LEFT, BOTH, DISABLED, NORMAL, X, Y, BOTTOM, RIGHT, HORIZONTAL, VERTICAL, NW, CENTER)
from tkinter import (Canvas, filedialog, messagebox, simpledialog, Toplevel, StringVar, Text,
                     colorchooser, DoubleVar)
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from datetime import datetime
import os
import time

# Importa módulos do sistema de banco de dados
try:
    # Quando importado como módulo
    from .database_manager import DatabaseManager
    from .model_selector import ModelSelectorDialog, SaveModelDialog
    from .utils import load_style_config, save_style_config, apply_style_config, get_style_config_path
    from .ml_classifier import MLSlotClassifier
except ImportError:
    # Quando executado diretamente
    try:
        from database_manager import DatabaseManager
        from model_selector import ModelSelectorDialog, SaveModelDialog
        from utils import load_style_config, save_style_config, apply_style_config, get_style_config_path
        from ml_classifier import MLSlotClassifier
    except ImportError:
        # Quando executado a partir do diretório raiz
        from modulos.database_manager import DatabaseManager
        from modulos.model_selector import ModelSelectorDialog, SaveModelDialog
        from modulos.utils import load_style_config, save_style_config, apply_style_config, get_style_config_path
        from modulos.ml_classifier import MLSlotClassifier

# ---------- parâmetros globais ------------------------------------------------
# Caminho para a pasta de modelos na raiz do projeto
# Usa caminhos relativos para permitir portabilidade
def get_project_root():
    """Retorna o caminho da raiz do projeto."""
    return Path(__file__).parent.parent

def get_model_dir():
    """Retorna o caminho para o diretório de modelos."""
    model_dir = get_project_root() / "modelos"
    model_dir.mkdir(exist_ok=True)
    return model_dir

def get_template_dir():
    """Retorna o caminho para o diretório de templates."""
    template_dir = get_model_dir() / "_templates"
    template_dir.mkdir(exist_ok=True)
    return template_dir

def get_model_template_dir(model_name, model_id):
    """Retorna o caminho para o diretório de templates de um modelo específico."""
    template_dir = get_project_root() / f"modelos/{model_name}_{model_id}/templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    return template_dir

# Define as variáveis globais como funções para obter os caminhos atualizados
MODEL_DIR = get_model_dir()
TEMPLATE_DIR = get_template_dir()

# Limiares de inspeção
THR_CORR = 0.1  # Limiar para template matching (clips)
MIN_PX = 10      # Contagem mínima de pixels para template matching (clips)

# Parâmetros do Canvas e Preview
PREVIEW_W = 1200  # Largura máxima do canvas para exibição inicial (aumentada)
PREVIEW_H = 900  # Altura máxima do canvas para exibição inicial (aumentada)

# Parâmetros ORB para registro de imagem
ORB_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Cores para desenho no canvas (Editor) - ALTERADAS PARA NOVO LAYOUT
COLOR_CLIP = "#FF6B6B"  # Vermelho coral
COLOR_SELECTED = "#FFE66D"  # Amarelo dourado
COLOR_DRAWING = "#A8E6CF"  # Verde claro

# Cores para desenho no canvas (Inspeção) - ALTERADAS PARA NOVO LAYOUT
COLOR_PASS = "#95E1D3"  # Verde menta
COLOR_FAIL = "#F38BA8"  # Rosa
COLOR_ALIGN_FAIL = "#FECA57"  # Laranja claro

# Caminho para o arquivo de configurações de estilo
STYLE_CONFIG_PATH = get_style_config_path()


# ---------- utilidades --------------------------------------------------------

def detect_cameras(max_cameras=5, callback=None):
    """
    Detecta webcams disponíveis no sistema.
    Retorna lista de índices de câmeras funcionais.
    Compatível com Windows e Raspberry Pi.
    
    Args:
        max_cameras: Número máximo de câmeras para testar
        callback: Função opcional a ser chamada após detecção com a lista de câmeras
    """
    available_cameras = []
    
    # Detecta o sistema operacional
    import platform
    is_windows = platform.system() == 'Windows'
    
    for i in range(max_cameras):
        try:
            # Usa DirectShow no Windows para evitar erros do obsensor
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if cap is not None and cap.isOpened():
                # Testa se consegue ler um frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
                cap.release()
        except Exception as e:
            # Silencia erros de câmeras não encontradas
            print(f"Erro ao testar câmera {i}: {e}")
            continue
    
    # Se não encontrar nenhuma câmera, adiciona índice 0 como padrão
    if not available_cameras:
        available_cameras.append(0)
        print("Nenhuma câmera detectada automaticamente. Usando índice 0 como padrão.")
    else:
        print(f"Câmeras detectadas: {available_cameras}")
    
    # Chama callback se fornecido
    if callback:
        callback(available_cameras)
    
    return available_cameras

def capture_image_from_camera(camera_index=0):
    """
    Captura uma única imagem da webcam especificada.
    Retorna a imagem capturada ou None em caso de erro.
    Compatível com Windows e Raspberry Pi.
    """
    try:
        # Detecta o sistema operacional
        import platform
        is_windows = platform.system() == 'Windows'
        
        # Usa DirectShow no Windows para melhor compatibilidade
        # No Raspberry Pi, usa a API padrão
        if is_windows:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
        if camera_index > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a câmera {camera_index}")
            return None
        
        # Aguarda um pouco para a câmera se estabilizar
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
        
        # Captura a imagem final
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None and frame.size > 0:
            print(f"Imagem capturada com sucesso da câmera {camera_index}")
            return frame
        else:
            print(f"Erro: Não foi possível capturar imagem da câmera {camera_index}")
            return None
            
    except Exception as e:
        print(f"Erro ao capturar imagem da câmera {camera_index}: {e}")
        return None

def cv2_to_tk(img_bgr, max_w=None, max_h=None, scale_percent=None):
    """
    Converte imagem OpenCV BGR para formato Tkinter PhotoImage,
    redimensionando para usar 100% da área disponível do canvas.
    
    Otimizada para preencher completamente o espaço disponível.
    """
    # Validação de entrada
    if img_bgr is None or img_bgr.size == 0:
        return None, 1.0
    
    h, w = img_bgr.shape[:2]
    scale = 1.0

    # Se scale_percent for fornecido, usa-o para calcular a escala
    if scale_percent is not None:
        scale = scale_percent / 100.0
    else:
        # Calcula escala para usar 100% da área disponível
        if max_w and max_h:
            # Calcula escalas para largura e altura
            scale_w = max_w / w
            scale_h = max_h / h
            # Usa a maior escala para preencher completamente o espaço
            scale = max(scale_w, scale_h)
        elif max_w:
            scale = max_w / w
        elif max_h:
            scale = max_h / h

    # Redimensiona para usar toda a área disponível
    if scale != 1.0:
        new_w = max(1, int(w * scale))  # Garante dimensão mínima
        new_h = max(1, int(h * scale))
        
        try:
            # Usa INTER_AREA para redução e INTER_LINEAR para ampliação
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)
        except cv2.error as e:
             print(f"Erro ao redimensionar imagem: {e}. Dimensões: ({new_w}x{new_h})")
             return None, 1.0
    else:
        img_bgr_resized = img_bgr

    # Conversão para Tkinter
    try:
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        photo_image = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        return photo_image, scale
    except Exception as e:
        print(f"Erro ao converter imagem para Tkinter: {e}")
        return None, scale




# Inicialização otimizada do detector ORB
try:
    # Configurações otimizadas para melhor performance
    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
        edgeThreshold=31,  # Reduz detecção em bordas para melhor performance
        firstLevel=0,      # Nível inicial da pirâmide
        WTA_K=2,          # Número de pontos para comparação
        scoreType=cv2.ORB_HARRIS_SCORE,  # Usa Harris score para melhor qualidade
        patchSize=31      # Tamanho do patch para descritores
    )
    print("Detector ORB inicializado com sucesso (configuração otimizada).")
except Exception as e:
    print(f"Erro ao inicializar ORB: {e}. O registro de imagem não funcionará.")
    orb = None

# Cache para descritores de imagem de referência (otimização)
_ref_image_cache = {
    'image_hash': None,
    'keypoints': None,
    'descriptors': None,
    'gray_image': None
}


def find_image_transform(img_ref, img_test):
    """
    Encontra a transformação entre duas imagens usando ORB.
    
    Otimizada com:
    - Cache para imagem de referência
    - Validação de entrada mais eficiente
    - Matching otimizado
    
    Retorna: (matriz_homografia, matches_count, error_message)
    """
    global _ref_image_cache
    
    if orb is None:
        error_msg = "Detector ORB não disponível."
        print(error_msg)
        return None, 0, error_msg
    
    # Validação de entrada otimizada
    if img_ref is None or img_test is None or img_ref.size == 0 or img_test.size == 0:
        error_msg = "Imagens de referência ou teste inválidas."
        print(error_msg)
        return None, 0, error_msg
    
    try:
        # Converte para escala de cinza
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY) if len(img_test.shape) == 3 else img_test
        
        # === CACHE PARA IMAGEM DE REFERÊNCIA ===
        # Calcula hash simples da imagem de referência
        ref_hash = hash(gray_ref.tobytes())
        
        # Verifica se pode usar cache
        if (_ref_image_cache['image_hash'] == ref_hash and 
            _ref_image_cache['keypoints'] is not None and 
            _ref_image_cache['descriptors'] is not None):
            # Usa dados do cache
            kp_ref = _ref_image_cache['keypoints']
            desc_ref = _ref_image_cache['descriptors']
            print("Usando cache para imagem de referência")
        else:
            # Detecta keypoints e descritores para referência
            kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
            # Atualiza cache
            _ref_image_cache.update({
                'image_hash': ref_hash,
                'keypoints': kp_ref,
                'descriptors': desc_ref,
                'gray_image': gray_ref.copy()
            })
            print("Cache atualizado para imagem de referência")
        
        # Detecta keypoints e descritores para teste (sempre novo)
        kp_test, desc_test = orb.detectAndCompute(gray_test, None)
        
        # Validação de descritores
        if desc_ref is None or desc_test is None:
            error_msg = "Não foi possível extrair descritores de uma das imagens."
            print(error_msg)
            return None, 0, error_msg
        
        if len(desc_ref) < 4 or len(desc_test) < 4:
            error_msg = f"Poucos descritores encontrados: ref={len(desc_ref)}, test={len(desc_test)}"
            print(error_msg)
            return None, 0, error_msg
        
        # === MATCHING OTIMIZADO ===
        # Usa BFMatcher otimizado com crossCheck
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc_ref, desc_test)
        
        if len(matches) < 4:
            error_msg = f"Poucos matches encontrados: {len(matches)}"
            print(error_msg)
            return None, len(matches), error_msg
        
        # Ordena matches por distância e filtra os melhores
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Usa apenas os melhores matches para melhor performance
        max_matches = min(len(matches), 100)  # Limita a 100 melhores matches
        good_matches = matches[:max_matches]
        
        # Extrai pontos correspondentes
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # === CÁLCULO DE HOMOGRAFIA OTIMIZADO ===
        # Usa parâmetros otimizados para RANSAC
        M, mask = cv2.findHomography(
            src_pts, dst_pts, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,  # Threshold mais restritivo
            maxIters=2000,              # Máximo de iterações
            confidence=0.99             # Confiança desejada
        )
        
        if M is None:
            error_msg = "Não foi possível calcular a homografia."
            print(error_msg)
            return None, len(good_matches), error_msg
        
        inliers_count = np.sum(mask)
        inlier_ratio = inliers_count / len(good_matches)
        
        print(f"Homografia calculada: {inliers_count}/{len(good_matches)} inliers ({inlier_ratio:.2%})")
        return M, inliers_count, None
        
    except Exception as e:
        error_msg = f"Erro em find_image_transform: {e}"
        print(error_msg)
        return None, 0, error_msg


def transform_rectangle(rect, M, img_shape):
    """
    Transforma um retângulo usando uma matriz de homografia.
    rect: (x, y, w, h)
    M: matriz de homografia 3x3
    img_shape: (height, width) da imagem de destino
    Retorna: (x, y, w, h) transformado ou None se inválido
    """
    if M is None:
        return None
    
    x, y, w, h = rect
    
    # Define os 4 cantos do retângulo
    corners = np.float32([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ]).reshape(-1, 1, 2)
    
    try:
        # Transforma os cantos
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        # Calcula o bounding box dos cantos transformados
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Garante que está dentro dos limites da imagem
        img_h, img_w = img_shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img_w, int(x_max))
        y_max = min(img_h, int(y_max))
        
        new_w = x_max - x_min
        new_h = y_max - y_min
        
        if new_w <= 0 or new_h <= 0:
            print(f"Retângulo transformado inválido: ({x_min}, {y_min}, {new_w}, {new_h})")
            return None
        
        return (x_min, y_min, new_w, new_h)
        
    except Exception as e:
        print(f"Erro ao transformar retângulo: {e}")
        return None


def check_slot(img_test, slot_data, M):
    """
    Verifica um slot na imagem de teste.
    Retorna: (passou, correlation, pixels, corners, bbox, log_msgs)
    """
    log_msgs = []
    corners = None
    bbox = [0, 0, 0, 0]
    
    try:
        slot_type = slot_data.get('tipo', 'clip')
        x, y, w, h = slot_data['x'], slot_data['y'], slot_data['w'], slot_data['h']
        
        # Calcula os cantos originais do slot
        original_corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        
        # Transforma o retângulo se temos matriz de homografia
        if M is not None:
            # Transforma os cantos usando a matriz de homografia
            corners_array = np.array(original_corners, dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_array, M)
            corners = [(int(pt[0][0]), int(pt[0][1])) for pt in transformed_corners]
            
            # Calcula bounding box dos cantos transformados
            x_coords = [pt[0] for pt in corners]
            y_coords = [pt[1] for pt in corners]
            x, y = max(0, min(x_coords)), max(0, min(y_coords))
            w = min(img_test.shape[1] - x, max(x_coords) - x)
            h = min(img_test.shape[0] - y, max(y_coords) - y)
            
            log_msgs.append(f"Slot transformado para ({x}, {y}, {w}, {h})")
        else:
            corners = original_corners
            log_msgs.append("Usando coordenadas originais (sem transformação)")
        
        bbox = [x, y, w, h]
        
        # Verifica se a ROI está dentro dos limites da imagem
        if x < 0 or y < 0 or x + w > img_test.shape[1] or y + h > img_test.shape[0]:
            log_msgs.append(f"ROI fora dos limites da imagem: ({x}, {y}, {w}, {h})")
            return False, 0.0, 0, corners, bbox, log_msgs
        
        # Extrai ROI
        roi = img_test[y:y+h, x:x+w]
        if roi.size == 0:
            log_msgs.append("ROI vazia")
            return False, 0.0, 0, corners, bbox, log_msgs
        
        if slot_type == 'clip':
            # Verifica se deve usar Machine Learning
            if slot_data.get('use_ml', False) and slot_data.get('ml_model_path'):
                try:
                    from .ml_classifier import MLSlotClassifier
                    
                    # Carrega o modelo ML
                    ml_classifier = MLSlotClassifier()
                    ml_classifier.load_model(slot_data['ml_model_path'])
                    
                    # Faz a predição usando ML
                    prediction, confidence = ml_classifier.predict(roi)
                    
                    # Converte predição para resultado booleano
                    is_ok = prediction == 1  # 1 = OK, 0 = NG
                    
                    log_msgs.append(f"ML: Predição={prediction} ({'OK' if is_ok else 'NG'}), Confiança={confidence:.3f}")
                    return is_ok, confidence, 0, corners, bbox, log_msgs
                    
                except Exception as ml_error:
                    log_msgs.append(f"Erro no ML, usando método tradicional: {str(ml_error)}")
                    # Continua com método tradicional em caso de erro
            
            # Verifica método de detecção
            detection_method = slot_data.get('detection_method', 'template_matching')
            
            if detection_method == 'histogram_analysis':
                # === ANÁLISE POR HISTOGRAMA ===
                try:
                    # Calcula histograma da ROI em HSV
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Parâmetros do histograma
                    h_bins = 50
                    s_bins = 60
                    hist_range = [0, 180, 0, 256]  # H: 0-179, S: 0-255
                    
                    # Calcula histograma 2D (H-S)
                    hist = cv2.calcHist([roi_hsv], [0, 1], None, [h_bins, s_bins], hist_range)
                    
                    # Normaliza histograma
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    # Calcula métricas do histograma
                    np.sum(hist)
                    np.mean(hist)
                    hist_std = np.std(hist)
                    hist_max = np.max(hist)
                    
                    # Calcula entropia do histograma
                    hist_flat = hist.flatten()
                    hist_flat = hist_flat[hist_flat > 0]  # Remove zeros
                    entropy = -np.sum(hist_flat * np.log2(hist_flat + 1e-10))
                    
                    # Score baseado em múltiplas métricas
                    # Combina entropia (diversidade de cores) e distribuição
                    entropy_score = min(entropy / 10.0, 1.0)  # Normaliza entropia
                    distribution_score = min(hist_std * 10, 1.0)  # Penaliza distribuições muito uniformes
                    intensity_score = min(hist_max * 2, 1.0)  # Considera picos de intensidade
                    
                    # Score final combinado
                    histogram_score = (entropy_score * 0.5 + distribution_score * 0.3 + intensity_score * 0.2)
                    
                    # Usa limiar personalizado do slot ou padrão
                    if 'correlation_threshold' in slot_data:
                        threshold = slot_data.get('correlation_threshold', 0.3)
                        threshold_source = "correlation_threshold"
                    else:
                        threshold = slot_data.get('detection_threshold', 30.0) / 100.0  # Converte % para decimal
                        threshold_source = "detection_threshold"
                    
                    # Usa a porcentagem para OK personalizada ou padrão
                    ok_threshold = slot_data.get('ok_threshold', 70) / 100.0  # Converte % para decimal
                    
                    # Verifica se passou baseado na porcentagem para OK
                    passou = histogram_score >= ok_threshold
                    
                    log_msgs.append(f"Histograma: {histogram_score:.3f} (limiar: {threshold:.2f} [{threshold_source}], % para OK: {ok_threshold:.2f}, entropia: {entropy:.2f}, std: {hist_std:.3f})")
                    return passou, histogram_score, 0, corners, bbox, log_msgs
                    
                except Exception as e:
                    log_msgs.append(f"Erro na análise por histograma: {str(e)}")
                    return False, 0.0, 0, corners, bbox, log_msgs
            
            elif detection_method == 'contour_analysis':
                # === ANÁLISE POR CONTORNO ===
                try:
                    # Converte para escala de cinza
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Aplica blur para reduzir ruído
                    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                    
                    # Detecta bordas com Canny
                    edges = cv2.Canny(roi_blur, 50, 150)
                    
                    # Encontra contornos
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Se não encontrou contornos, retorna falha
                    if not contours:
                        log_msgs.append("Nenhum contorno encontrado")
                        return False, 0.0, 0, corners, bbox, log_msgs
                    
                    # Calcula área total da ROI
                    roi_area = roi.shape[0] * roi.shape[1]
                    
                    # Calcula área total dos contornos
                    contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
                    
                    # Calcula número de contornos
                    num_contours = len(contours)
                    
                    # Calcula perímetro total
                    total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
                    
                    # Calcula complexidade média dos contornos (razão perímetro/área)
                    complexity = total_perimeter / (contour_area + 1e-10)
                    
                    # Normaliza métricas
                    area_ratio = min(contour_area / roi_area, 1.0)  # Razão entre área de contornos e área total
                    contour_count_score = min(num_contours / 10.0, 1.0)  # Normaliza contagem de contornos
                    complexity_score = min(1.0, 1.0 / (complexity + 0.1))  # Inverte para que menor complexidade = maior score
                    
                    # Score final combinado
                    contour_score = (area_ratio * 0.4 + contour_count_score * 0.3 + complexity_score * 0.3)
                    
                    # Usa limiar personalizado do slot ou padrão
                    threshold = slot_data.get('detection_threshold', 0.5)
                    
                    # Usa a porcentagem para OK personalizada ou padrão
                    ok_threshold = slot_data.get('ok_threshold', 70) / 100.0  # Converte % para decimal
                    
                    # Verifica se passou baseado na porcentagem para OK
                    passou = contour_score >= ok_threshold
                    
                    log_msgs.append(f"Contorno: {contour_score:.3f} (limiar: {threshold:.2f}, % para OK: {ok_threshold:.2f}, contornos: {num_contours}, área: {area_ratio:.2f}, complexidade: {complexity:.2f})")
                    return passou, contour_score, 0, corners, bbox, log_msgs
                    
                except Exception as e:
                    log_msgs.append(f"Erro na análise por contorno: {str(e)}")
                    return False, 0.0, 0, corners, bbox, log_msgs
            
            elif detection_method == 'image_comparison':
                # === COMPARAÇÃO DIRETA DE IMAGEM ===
                try:
                    template_path = slot_data.get('template_path')
                    if not template_path or not Path(template_path).exists():
                        log_msgs.append("Template não encontrado para comparação de imagem")
                        return False, 0.0, 0, corners, bbox, log_msgs
                    
                    # Carrega o template
                    template = cv2.imread(str(template_path))
                    if template is None:
                        log_msgs.append("Erro ao carregar template para comparação de imagem")
                        return False, 0.0, 0, corners, bbox, log_msgs
                    
                    # Redimensiona o template para o tamanho da ROI
                    template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                    
                    # Converte para escala de cinza
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Calcula SSIM (Structural Similarity Index)
                    from skimage.metrics import structural_similarity as ssim
                    try:
                        ssim_score, _ = ssim(roi_gray, template_gray, full=True)
                    except ImportError:
                        # Fallback se skimage não estiver disponível
                        # Calcula MSE (Mean Squared Error) e converte para similaridade
                        mse = np.mean((roi_gray.astype("float") - template_gray.astype("float")) ** 2)
                        ssim_score = 1 - (mse / 255**2)  # Normaliza para [0,1] onde 1 é perfeito
                    
                    # Calcula diferença absoluta
                    diff = cv2.absdiff(roi_gray, template_gray)
                    diff_score = 1.0 - (np.mean(diff) / 255.0)  # Normaliza para [0,1] onde 1 é perfeito
                    
                    # Calcula histogramas e compara
                    hist_roi = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
                    hist_template = cv2.calcHist([template_gray], [0], None, [256], [0, 256])
                    cv2.normalize(hist_roi, hist_roi, 0, 1, cv2.NORM_MINMAX)
                    cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
                    hist_score = cv2.compareHist(hist_roi, hist_template, cv2.HISTCMP_CORREL)
                    
                    # Score final combinado
                    comparison_score = (ssim_score * 0.5 + diff_score * 0.3 + hist_score * 0.2)
                    
                    # Usa limiar personalizado do slot ou padrão
                    threshold = slot_data.get('detection_threshold', 0.7)
                    
                    # Usa a porcentagem para OK personalizada ou padrão
                    ok_threshold = slot_data.get('ok_threshold', 70) / 100.0  # Converte % para decimal
                    
                    # Verifica se passou baseado na porcentagem para OK
                    passou = comparison_score >= ok_threshold
                    
                    log_msgs.append(f"Comparação: {comparison_score:.3f} (limiar: {threshold:.2f}, % para OK: {ok_threshold:.2f}, SSIM: {ssim_score:.2f}, Diff: {diff_score:.2f}, Hist: {hist_score:.2f})")
                    return passou, comparison_score, 0, corners, bbox, log_msgs
                    
                except Exception as e:
                    log_msgs.append(f"Erro na comparação de imagem: {str(e)}")
                    return False, 0.0, 0, corners, bbox, log_msgs
            
            else:  # template_matching (método padrão)
                # === TEMPLATE MATCHING PARA CLIPS ===
                template_path = slot_data.get('template_path')
                if not template_path or not Path(template_path).exists():
                    log_msgs.append("Template não encontrado")
                    return False, 0.0, 0, corners, bbox, log_msgs
                
                template = cv2.imread(str(template_path))
                if template is None:
                    log_msgs.append("Erro ao carregar template")
                    return False, 0.0, 0, corners, bbox, log_msgs
                
                # === TEMPLATE MATCHING OTIMIZADO ===
                slot_data.get('correlation_threshold', 0.7)
                template_method_str = slot_data.get('template_method', 'TM_CCOEFF_NORMED')
                scale_tolerance = slot_data.get('scale_tolerance', 10.0) / 100.0
                
                # Mapeamento otimizado de métodos
                method_map = {
                    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
                    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
                    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
                }
                template_method = method_map.get(template_method_str, cv2.TM_CCOEFF_NORMED)
                
                max_val = 0.0
                best_scale = 1.0
                
                # Otimização: reduz número de escalas testadas
                if scale_tolerance > 0:
                    # Testa apenas 3 escalas para melhor performance
                    scales = [1.0 - scale_tolerance, 1.0, 1.0 + scale_tolerance]
                else:
                    scales = [1.0]  # Apenas escala original
                
                # Pré-converte template para escala de cinza se necessário (otimização)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            for scale in scales:
                # Calcula dimensões da escala
                scaled_w = int(template_gray.shape[1] * scale)
                scaled_h = int(template_gray.shape[0] * scale)
                
                # Validação de dimensões otimizada
                if (scaled_w <= 0 or scaled_h <= 0 or 
                    scaled_w > roi_gray.shape[1] or scaled_h > roi_gray.shape[0]):
                    continue
                
                # Redimensiona template (usa INTER_AREA para redução, INTER_LINEAR para ampliação)
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=interpolation)
                
                # Template matching otimizado (usa imagens em escala de cinza)
                result = cv2.matchTemplate(roi_gray, scaled_template, template_method)
                
                # Extrai valor de correlação
                if template_method == cv2.TM_SQDIFF_NORMED:
                    min_val, _, _, _ = cv2.minMaxLoc(result)
                    current_val = 1.0 - min_val  # Inverte para SQDIFF
                else:
                    _, current_val, _, _ = cv2.minMaxLoc(result)
                
                # Atualiza melhor resultado
                if current_val > max_val:
                    max_val = current_val
                    best_scale = scale
            
            # Usa limiar personalizado do slot ou padrão
            # Prioridade: correlation_threshold > detection_threshold > padrão global
            if 'correlation_threshold' in slot_data:
                threshold = slot_data.get('correlation_threshold', 0.1)
                threshold_source = "correlation_threshold"
            else:
                threshold = slot_data.get('detection_threshold', 70.0) / 100.0  # Converte % para decimal
                threshold_source = "detection_threshold"
            
            # Usa a porcentagem para OK personalizada ou padrão
            ok_threshold = slot_data.get('ok_threshold', 70) / 100.0  # Converte % para decimal
            
            # Verifica se passou baseado na porcentagem para OK
            passou = max_val >= ok_threshold
            
            log_msgs.append(f"Correlação: {max_val:.3f} (limiar: {threshold:.2f} [{threshold_source}], % para OK: {ok_threshold:.2f}, escala: {best_scale:.2f}, método: {template_method_str})")
            return passou, max_val, 0, corners, bbox, log_msgs
        
        else:  # fita - tipo removido, apenas clips são suportados
            log_msgs.append("Tipo 'fita' não é mais suportado - apenas template matching para 'clip'")
            return False, 0.0, 0, corners, bbox, log_msgs
    
    except Exception as e:
        log_msgs.append(f"Erro: {str(e)}")
        print(f"Erro em check_slot: {e}")
        return False, 0.0, 0, corners, bbox, log_msgs


class EditSlotDialog(Toplevel):
    def __init__(self, parent, slot_data, malha_frame_instance):
        """Inicializa diálogo de edição com configuração otimizada"""
        try:
            super().__init__(parent)
            
            # Verifica se os parâmetros são válidos
            if not parent or not slot_data or not malha_frame_instance:
                raise ValueError("Parâmetros inválidos para EditSlotDialog")
            
            # Verifica se o slot_data tem as chaves necessárias
            basic_keys = ['id', 'x', 'y', 'w', 'h', 'tipo']
            required_keys = basic_keys.copy()
            
            # Para slots do tipo 'clip', adiciona campos específicos
            if slot_data.get('tipo') == 'clip':
                clip_keys = ['cor', 'detection_threshold']
                required_keys.extend(clip_keys)
            
            missing_keys = [key for key in required_keys if key not in slot_data]
            if missing_keys:
                raise ValueError(f"Dados do slot incompletos. Chaves ausentes: {missing_keys}")
            
            # === INICIALIZAÇÃO DE DADOS ===
            self.slot_data = slot_data.copy()
            self.malha_frame = malha_frame_instance
            self.result = None
            self._is_destroyed = False
            
            # Inicializa configurações de estilo se não existirem
            if 'style_config' not in self.slot_data:
                self.slot_data['style_config'] = {
                    'bg_color': '#1E1E1E',  # Cor de fundo padrão
                    'text_color': '#FFFFFF',  # Cor do texto padrão
                    'ok_color': '#95E1D3',  # Cor para OK
                    'ng_color': '#F38BA8',  # Cor para NG
                    'selection_color': '#FFE66D',  # Cor de seleção
                    'ok_font': 'Arial 12 bold',  # Fonte para OK
                    'ng_font': 'Arial 12 bold'   # Fonte para NG
                }
            
            # === CONFIGURAÇÃO DA JANELA ===
            self.title(f"Editando Slot {slot_data['id']}")
            self.geometry("400x650")
            self.resizable(False, False)
            self.configure(bg='#2E2E2E')  # Cor de fundo escura para toda a janela
            
            # Configuração modal otimizada
            self.transient(parent)
            self.protocol("WM_DELETE_WINDOW", self.cancel)
            
            # Verifica se a janela pai ainda existe
            if not parent.winfo_exists():
                raise RuntimeError("Janela pai não existe mais")
            
            try:
                # === CONFIGURAÇÃO DA INTERFACE ===
                print("Iniciando setup_ui...")
                self.setup_ui()
                print("setup_ui concluído")
                
                print("Iniciando load_slot_data...")
                self.load_slot_data()
                print("load_slot_data concluído")
                
                print("Iniciando center_window...")
                self.center_window()
                print("center_window concluído")
                
                # Aplica modalidade diretamente
                print("Aplicando modalidade...")
                self.apply_modal_grab()
                print("Modalidade aplicada")
                
            except Exception as ui_error:
                print(f"Erro ao configurar interface: {ui_error}")
                import traceback
                traceback.print_exc()
                raise ui_error
                
        except Exception as e:
            print(f"Erro ao inicializar EditSlotDialog: {e}")
            import traceback
            traceback.print_exc()
            
            # Tenta mostrar erro se possível
            try:
                messagebox.showerror("Erro", f"Erro ao abrir editor: {str(e)}")
            except:
                print("Não foi possível mostrar messagebox de erro")
            
            # Destrói a janela se foi criada
            try:
                if hasattr(self, 'winfo_exists') and self.winfo_exists():
                    self.destroy()
            except:
                pass
            
            # Re-levanta a exceção para que o chamador saiba que houve erro
            raise e
    
    def apply_modal_grab(self):
        """Aplica grab_set() após a janela estar completamente inicializada"""
        try:
            # Temporariamente removendo grab_set() para evitar travamentos
            # self.grab_set()
            self.focus_set()
            print("Modal grab aplicado com sucesso (sem grab_set)")
        except Exception as e:
            print(f"Erro ao aplicar modal grab: {e}")
    
    def center_window(self):
        try:
            print("Iniciando centralização da janela...")
            self.update_idletasks()
            
            # Centralização direta sem delay
            width = 500  # largura padrão
            height = 400  # altura padrão
            
            x = (self.winfo_screenwidth() // 2) - (width // 2)
            y = (self.winfo_screenheight() // 2) - (height // 2)
            
            self.geometry(f"{width}x{height}+{x}+{y}")
            print(f"Janela centralizada: {width}x{height}+{x}+{y}")
        except Exception as e:
            print(f"Erro ao centralizar janela: {e}")
    
    def setup_ui(self):
        """Configura interface otimizada do diálogo de edição"""
        try:
            print("Criando frame principal...")
            # === FRAME PRINCIPAL ===
            main_frame = ttk.Frame(self)
            main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
            print("Frame principal criado")
            
            print("Criando seção de informações...")
            # === INFORMAÇÕES DO SLOT ===
            info_frame = ttk.LabelFrame(main_frame, text="Informações do Slot")
            info_frame.pack(fill=X, pady=(0, 10))
            
            # Labels de informação otimizadas
            slot_info = f"ID: {self.slot_data['id']} | Tipo: {self.slot_data['tipo']}"
            ttk.Label(info_frame, text=slot_info).pack(anchor="w", padx=5, pady=5)
            print("Seção de informações criada")
            
            print("Criando seção de edição de malha...")
            # === EDIÇÃO DE MALHA ===
            mesh_frame = ttk.LabelFrame(main_frame, text="Posição e Dimensões")
            mesh_frame.pack(fill=X, pady=(0, 10))
            
            # Inicialização otimizada de variáveis
            print("Inicializando variáveis...")
            self.x_var = StringVar()
            self.y_var = StringVar()
            self.w_var = StringVar()
            self.h_var = StringVar()
            self.detection_threshold_var = StringVar()
            print("Variáveis inicializadas")
            
            # Criação de um frame com grid de 2 colunas para melhor organização
            mesh_grid = ttk.Frame(mesh_frame)
            mesh_grid.pack(fill=X, padx=10, pady=10)
            mesh_grid.columnconfigure(0, weight=1)
            mesh_grid.columnconfigure(1, weight=1)
            
            print("Criando campos de entrada...")
            # Primeira linha: X e Y lado a lado
            ttk.Label(mesh_grid, text="Posição X:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            ttk.Entry(mesh_grid, textvariable=self.x_var, width=8).grid(row=0, column=0, sticky="e", padx=5, pady=5)
            
            ttk.Label(mesh_grid, text="Posição Y:").grid(row=0, column=1, sticky="w", padx=5, pady=5)
            ttk.Entry(mesh_grid, textvariable=self.y_var, width=8).grid(row=0, column=1, sticky="e", padx=5, pady=5)
            
            # Segunda linha: Largura e Altura lado a lado
            ttk.Label(mesh_grid, text="Largura:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            ttk.Entry(mesh_grid, textvariable=self.w_var, width=8).grid(row=1, column=0, sticky="e", padx=5, pady=5)
            
            ttk.Label(mesh_grid, text="Altura:").grid(row=1, column=1, sticky="w", padx=5, pady=5)
            ttk.Entry(mesh_grid, textvariable=self.h_var, width=8).grid(row=1, column=1, sticky="e", padx=5, pady=5)
            print("Campos de entrada criados")
            
            print("Criando seção de configurações...")
            # === CONFIGURAÇÕES BÁSICAS ===
            config_frame = ttk.LabelFrame(main_frame, text="Configurações")
            config_frame.pack(fill=X, pady=(0, 10))
            
            # Campo de limiar otimizado
            threshold_frame = ttk.Frame(config_frame)
            threshold_frame.pack(fill=X, padx=5, pady=5)
            
            ttk.Label(threshold_frame, text="Limiar de Detecção (%):").pack(side=LEFT)
            ttk.Entry(threshold_frame, textvariable=self.detection_threshold_var, width=10).pack(side=LEFT, padx=(5, 0))
            print("Seção de configurações criada")
            
            print("Criando botões de ação...")
            # === BOTÕES DE AÇÃO ===
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=X, pady=(10, 0))
            
            # Botões otimizados
            ttk.Button(button_frame, text="Salvar", command=self.save_changes).pack(side=LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT)
            print("Botões de ação criados")
        
            print("setup_ui concluído com sucesso")
            
        except Exception as e:
            print(f"Erro na configuração da UI: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Falha ao configurar interface: {e}")
            self.destroy()
    
    def load_slot_data(self):
        try:
            print(f"Carregando dados do slot: {self.slot_data}")
            # Carrega os dados do slot nos campos da interface
            if self.slot_data:
                self.x_var.set(str(self.slot_data.get('x', 0)))
                self.y_var.set(str(self.slot_data.get('y', 0)))
                self.w_var.set(str(self.slot_data.get('w', 100)))
                self.h_var.set(str(self.slot_data.get('h', 100)))
                self.detection_threshold_var.set(str(self.slot_data.get('detection_threshold', 50)))
                print("Dados carregados com sucesso nos campos")
            else:
                print("Nenhum dado de slot para carregar")
        except Exception as e:
            print(f"Erro ao carregar dados do slot: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar dados do slot: {str(e)}")
    
    def get_hex_color(self, bgr):
        """Converte cor BGR para hexadecimal."""
        try:
            b, g, r = bgr
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return "#000000"
    
    def update_template_visibility(self, event=None):
        """Mostra ou oculta o template do clip."""
        if self.slot_data['tipo'] != 'clip':
            return
        
        template_path = self.slot_data.get('template_path')
        if not template_path or not Path(template_path).exists():
            messagebox.showwarning("Aviso", "Template não encontrado.")
            return
        
        if self.show_template_var.get():
            # Mostra template
            template = cv2.imread(str(template_path))
            if template is not None:
                cv2.imshow(f"Template - Slot {self.slot_data['id']}", template)
        else:
            # Oculta template
            cv2.destroyWindow(f"Template - Slot {self.slot_data['id']}")
    
    def pick_new_color(self):
        """Função simplificada para escolher nova cor."""
        try:
            messagebox.showinfo("Info", "Função de seleção de cor simplificada.")
        except Exception as e:
            print(f"Erro na seleção de cor: {e}")
            messagebox.showerror("Erro", f"Erro na seleção de cor: {str(e)}")
    
    def save_changes(self):
        """Salva as alterações feitas no slot."""
        try:
            print(f"\n=== SALVANDO ALTERAÇÕES DO SLOT {self.slot_data.get('id', 'N/A')} ===")
            
            # Validação e conversão dos valores
            try:
                x_val = int(self.x_var.get().strip())
                y_val = int(self.y_var.get().strip())
                w_val = int(self.w_var.get().strip())
                h_val = int(self.h_var.get().strip())
                threshold_val = float(self.detection_threshold_var.get().strip())
                
                # Validações básicas
                if w_val <= 0 or h_val <= 0:
                    raise ValueError("Largura e altura devem ser maiores que zero")
                if threshold_val < 0 or threshold_val > 100:
                    raise ValueError("Limiar deve estar entre 0 e 100")
                    
            except ValueError as ve:
                messagebox.showerror("Erro de Validação", f"Valores inválidos: {str(ve)}")
                return
            
            # Salva alterações de malha (posição e tamanho)
            self.slot_data['x'] = x_val
            self.slot_data['y'] = y_val
            self.slot_data['w'] = w_val
            self.slot_data['h'] = h_val
            
            # Salva limiar de detecção
            self.slot_data['detection_threshold'] = threshold_val
            
            print(f"Dados salvos: posição ({self.slot_data['x']},{self.slot_data['y']}), tamanho ({self.slot_data['w']},{self.slot_data['h']}), limiar {self.slot_data['detection_threshold']}%")
            
            # Chama o método update_slot_data da instância malha_frame
            print("Chamando update_slot_data...")
            self.malha_frame.update_slot_data(self.slot_data)
            print("Alterações salvas com sucesso!")
            
            self.destroy()
        
        except Exception as e:
            print(f"ERRO ao salvar alterações: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
    def cancel(self):
        """Cancela a edição com proteções contra travamentos."""
        try:
            print("Cancelando edição...")
            
            # Verifica se a janela já foi destruída
            if hasattr(self, '_is_destroyed') and self._is_destroyed:
                print("Janela já foi destruída anteriormente")
                return
            
            # Marca como destruída para evitar múltiplas chamadas
            self._is_destroyed = True
            
            # Remove grab modal se estiver ativo
            try:
                self.grab_release()
            except:
                pass
            
            # Fecha janela de template se estiver aberta
            if hasattr(self, 'slot_data') and self.slot_data and self.slot_data.get('tipo') == 'clip':
                try:
                    cv2.destroyWindow(f"Template - Slot {self.slot_data['id']}")
                except:
                    pass  # Ignora erro se janela não existir
            
            # Limpa referências
            self.result = None
            if hasattr(self, 'malha_frame'):
                self.malha_frame = None
            
            print("Destruindo janela...")
            
            # Verifica se a janela ainda existe antes de destruir
            if self.winfo_exists():
                self.destroy()
                print("Janela destruída com sucesso")
            else:
                print("Janela já foi destruída")                
        except Exception as e:
            print(f"Erro ao cancelar: {e}")
            import traceback
            traceback.print_exc()
            
            # Tentativa final de destruir a janela
            try:
                if hasattr(self, 'winfo_exists') and self.winfo_exists():
                    self.destroy()
            except:
                print("Não foi possível destruir a janela na tentativa final")


class SlotTrainingDialog(Toplevel):
    """Diálogo para treinamento de slots com feedback OK/NG."""
    
    def __init__(self, parent, slot_data, montagem_instance):
        super().__init__(parent)
        self.slot_data = slot_data
        self.montagem_instance = montagem_instance
        self.training_samples = []  # Lista de amostras de treinamento
        
        # Inicializa classificador ML
        self.ml_classifier = MLSlotClassifier(slot_id=str(slot_data['id']))
        self.use_ml = False  # Flag para usar ML ou método tradicional
        
        # Define o diretório para salvar as amostras
        template_path = self.slot_data.get('template_path')
        if template_path:
            template_dir = os.path.dirname(template_path)
            self.samples_dir = os.path.join(template_dir, f"slot_{slot_data['id']}_samples")
        else:
            # Cria diretório padrão se template_path não estiver definido
            # Usa o diretório do projeto como base
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            samples_base_dir = os.path.join(project_dir, "modelos", "_samples")
            self.samples_dir = os.path.join(samples_base_dir, f"slot_{slot_data['id']}_samples")
            print(f"AVISO: template_path não definido. Usando diretório padrão: {self.samples_dir}")
            
        # Cria diretórios se não existirem
        try:
            os.makedirs(os.path.join(self.samples_dir, "ok"), exist_ok=True)
            os.makedirs(os.path.join(self.samples_dir, "ng"), exist_ok=True)
        except Exception as e:
            print(f"Erro ao criar diretórios de amostras: {e}")
            self.samples_dir = None
        
        self.title(f"Treinamento - Slot {slot_data['id']}")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Variáveis
        self.current_image = None
        self.current_roi = None
        
        self.setup_ui()
        self.center_window()
        self.apply_modal_grab()
        
    def apply_modal_grab(self):
        """Aplica grab modal para manter foco na janela."""
        self.transient(self.master)
        self.grab_set()
        
    def center_window(self):
        """Centraliza a janela na tela."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_ui(self):
        """Configura a interface do diálogo de treinamento."""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text=f"🎯 Treinamento do Slot {self.slot_data['id']}", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Frame superior - controles
        controls_frame = ttk.LabelFrame(main_frame, text="📷 Controles de Captura")
        controls_frame.pack(fill=X, pady=(0, 10))
        
        # Seleção do método de treinamento
        method_frame = ttk.Frame(controls_frame)
        method_frame.pack(fill=X, padx=10, pady=(10, 5))
        
        ttk.Label(method_frame, text="🤖 Método de Treinamento:", font=("Arial", 10, "bold")).pack(side=LEFT)
        
        self.training_method_var = StringVar(value="traditional")
        self.radio_traditional = ttk.Radiobutton(method_frame, text="Tradicional (Threshold)", 
                                               variable=self.training_method_var, value="traditional",
                                               command=self.on_method_change)
        self.radio_traditional.pack(side=LEFT, padx=(10, 5))
        
        self.radio_ml = ttk.Radiobutton(method_frame, text="Machine Learning (Scikit-Learn)", 
                                      variable=self.training_method_var, value="ml",
                                      command=self.on_method_change)
        self.radio_ml.pack(side=LEFT, padx=(5, 0))
        
        # Botões de captura
        capture_frame = ttk.Frame(controls_frame)
        capture_frame.pack(fill=X, padx=10, pady=10)
        
        self.btn_capture_webcam = ttk.Button(capture_frame, text="📷 Capturar da Webcam", 
                                           command=self.capture_from_webcam, width=20)
        self.btn_capture_webcam.pack(side=LEFT, padx=(0, 10))
        
        self.btn_load_image = ttk.Button(capture_frame, text="📁 Carregar Imagem", 
                                       command=self.load_image_file, width=20)
        self.btn_load_image.pack(side=LEFT, padx=(0, 10))
        
        # Botão para limpar histórico
        self.btn_clear_history = ttk.Button(capture_frame, text="🗑️ Limpar Histórico", 
                                          command=self.clear_training_history, width=20)
        self.btn_clear_history.pack(side=RIGHT)
        
        # Frame central dividido em duas colunas
        central_frame = ttk.Frame(main_frame)
        central_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Coluna esquerda - visualização atual
        left_frame = ttk.LabelFrame(central_frame, text="🖼️ Visualização Atual")
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        # Canvas para exibir imagem atual
        self.canvas = Canvas(left_frame, bg="#1E1E1E")
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Botões de feedback para imagem atual
        feedback_buttons_frame = ttk.Frame(left_frame)
        feedback_buttons_frame.pack(fill=X, padx=10, pady=(0, 10))
        
        self.btn_mark_ok = ttk.Button(feedback_buttons_frame, text="✅ Marcar como OK", 
                                    command=self.mark_as_ok, state=DISABLED, width=15)
        self.btn_mark_ok.pack(side=LEFT, padx=(0, 10))
        
        self.btn_mark_ng = ttk.Button(feedback_buttons_frame, text="❌ Marcar como NG", 
                                    command=self.mark_as_ng, state=DISABLED, width=15)
        self.btn_mark_ng.pack(side=LEFT)
        
        # Coluna direita - histórico de treinamento
        right_frame = ttk.LabelFrame(central_frame, text="📊 Histórico de Treinamento")
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        # Notebook para separar OK e NG
        self.history_notebook = ttk.Notebook(right_frame)
        self.history_notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Aba OK
        self.ok_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ok_frame, text="✅ Amostras OK (0)")
        
        # Scrollable frame para amostras OK
        self.ok_canvas = Canvas(self.ok_frame, bg="#f0f8f0")
        self.ok_scrollbar = ttk.Scrollbar(self.ok_frame, orient="vertical", command=self.ok_canvas.yview)
        self.ok_scrollable_frame = ttk.Frame(self.ok_canvas)
        
        self.ok_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ok_canvas.configure(scrollregion=self.ok_canvas.bbox("all"))
        )
        
        self.ok_canvas.create_window((0, 0), window=self.ok_scrollable_frame, anchor="nw")
        self.ok_canvas.configure(yscrollcommand=self.ok_scrollbar.set)
        
        self.ok_canvas.pack(side="left", fill="both", expand=True)
        self.ok_scrollbar.pack(side="right", fill="y")
        
        # Adiciona suporte para scroll com mouse wheel
        self.ok_canvas.bind("<MouseWheel>", lambda e: self.ok_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Aba NG
        self.ng_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ng_frame, text="❌ Amostras NG (0)")
        
        # Scrollable frame para amostras NG
        self.ng_canvas = Canvas(self.ng_frame, bg="#f8f0f0")
        self.ng_scrollbar = ttk.Scrollbar(self.ng_frame, orient="vertical", command=self.ng_canvas.yview)
        self.ng_scrollable_frame = ttk.Frame(self.ng_canvas)
        
        self.ng_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ng_canvas.configure(scrollregion=self.ng_canvas.bbox("all"))
        )
        
        self.ng_canvas.create_window((0, 0), window=self.ng_scrollable_frame, anchor="nw")
        self.ng_canvas.configure(yscrollcommand=self.ng_scrollbar.set)
        
        self.ng_canvas.pack(side="left", fill="both", expand=True)
        self.ng_scrollbar.pack(side="right", fill="y")
        
        # Adiciona suporte para scroll com mouse wheel
        self.ng_canvas.bind("<MouseWheel>", lambda e: self.ng_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        # Frame inferior - informações e ações
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=X, pady=(10, 0))
        
        # Informações de treinamento
        info_frame = ttk.LabelFrame(bottom_frame, text="📈 Estatísticas")
        info_frame.pack(fill=X, pady=(0, 10))
        
        stats_frame = ttk.Frame(info_frame)
        stats_frame.pack(fill=X, padx=10, pady=10)
        
        self.info_label = ttk.Label(stats_frame, text="Amostras coletadas: 0 OK, 0 NG", 
                                   font=("Arial", 10, "bold"))
        self.info_label.pack(side=LEFT)
        
        self.threshold_label = ttk.Label(stats_frame, text="Threshold atual: N/A", 
                                        font=("Arial", 10))
        self.threshold_label.pack(side=RIGHT)
        
        # Botões de ação
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(fill=X)
        
        self.btn_apply_training = ttk.Button(action_frame, text="🚀 Aplicar Treinamento", 
                                           command=self.apply_training, state=DISABLED, width=20)
        self.btn_apply_training.pack(side=LEFT, padx=(0, 10))
        
        # Botão para treinar ML (inicialmente oculto)
        self.btn_train_ml = ttk.Button(action_frame, text="🤖 Treinar ML", 
                                     command=self.train_ml_model, state=DISABLED, width=15)
        self.btn_train_ml.pack(side=LEFT, padx=(0, 10))
        self.btn_train_ml.pack_forget()  # Oculta inicialmente
        
        # Botão para salvar modelo ML
        self.btn_save_ml = ttk.Button(action_frame, text="💾 Salvar Modelo ML", 
                                    command=self.save_ml_model, state=DISABLED, width=18)
        self.btn_save_ml.pack(side=LEFT, padx=(0, 10))
        self.btn_save_ml.pack_forget()  # Oculta inicialmente
        
        self.btn_cancel = ttk.Button(action_frame, text="❌ Cancelar", 
                                   command=self.cancel, width=15)
        self.btn_cancel.pack(side=RIGHT)
        
        # Atualiza threshold atual se existir
        current_threshold = self.slot_data.get('correlation_threshold', 
                                             self.slot_data.get('detection_threshold', 'N/A'))
        if current_threshold != 'N/A':
            self.threshold_label.config(text=f"Threshold atual: {current_threshold:.3f}")
        
        # Carrega amostras existentes se houver
        self.load_existing_samples()
        
    def capture_from_webcam(self):
        """Captura imagem da webcam para treinamento."""
        try:
            # Usa a mesma função de captura da montagem
            camera_index = 0
            if hasattr(self.montagem_instance, 'camera_combo') and self.montagem_instance.camera_combo.get():
                camera_index = int(self.montagem_instance.camera_combo.get())
            
            captured_image = capture_image_from_camera(camera_index)
            
            if captured_image is not None:
                self.process_captured_image(captured_image)
            else:
                messagebox.showerror("Erro", "Falha ao capturar imagem da webcam.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
            
    def load_image_file(self):
        """Carrega imagem de arquivo para treinamento."""
        try:
            file_path = filedialog.askopenfilename(
                title="Selecionar Imagem",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            
            if file_path:
                image = cv2.imread(file_path)
                if image is not None:
                    self.process_captured_image(image)
                else:
                    messagebox.showerror("Erro", "Falha ao carregar a imagem.")
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
            
    def process_captured_image(self, image):
        """Processa a imagem capturada e extrai a ROI do slot."""
        try:
            self.current_image = image.copy()
            
            # Encontra a transformação entre a imagem de referência e a capturada
            if not hasattr(self.montagem_instance, 'img_original') or self.montagem_instance.img_original is None:
                messagebox.showerror("Erro", "Imagem de referência não carregada.")
                return
                
            M, inliers_count, error_msg = find_image_transform(self.montagem_instance.img_original, image)
            
            if M is None:
                messagebox.showwarning("Aviso", "Não foi possível alinhar a imagem. Usando coordenadas diretas.")
                # Usa coordenadas diretas se não conseguir alinhar
                x, y, w, h = self.slot_data['x'], self.slot_data['y'], self.slot_data['w'], self.slot_data['h']
            else:
                # Transforma as coordenadas do slot
                original_corners = np.array([[
                    [self.slot_data['x'], self.slot_data['y']], 
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y']],
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y'] + self.slot_data['h']],
                    [self.slot_data['x'], self.slot_data['y'] + self.slot_data['h']]
                ]], dtype=np.float32)
                
                transformed_corners = cv2.perspectiveTransform(original_corners, M)[0]
                
                # Calcula bounding box
                x = int(min(corner[0] for corner in transformed_corners))
                y = int(min(corner[1] for corner in transformed_corners))
                w = int(max(corner[0] for corner in transformed_corners) - x)
                h = int(max(corner[1] for corner in transformed_corners) - y)
            
            # Valida e ajusta coordenadas
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                messagebox.showerror("Erro", "ROI inválida detectada.")
                return
                
            # Extrai ROI
            self.current_roi = image[y:y+h, x:x+w].copy()
            
            # Exibe a imagem com a ROI destacada
            self.display_image_with_roi(image, x, y, w, h)
            
            # Habilita botões de feedback
            self.btn_mark_ok.config(state=NORMAL)
            self.btn_mark_ng.config(state=NORMAL)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")
            
    def display_image_with_roi(self, image, roi_x, roi_y, roi_w, roi_h):
        """Exibe a imagem com a ROI destacada no canvas."""
        try:
            # Cria cópia da imagem para desenhar
            display_image = image.copy()
            
            # Desenha retângulo da ROI
            cv2.rectangle(display_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
            
            # === AJUSTE AUTOMÁTICO AO CANVAS ===
            try:
                # Obtém o tamanho atual do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # Se o canvas ainda não foi renderizado, use valores padrão
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 640
                    canvas_height = 480
            except Exception as canvas_error:
                print(f"Erro ao obter dimensões do canvas: {canvas_error}")
                canvas_width = 640
                canvas_height = 480
            
            # Converte para exibição no canvas
            tk_image, _ = cv2_to_tk(display_image, max_w=canvas_width, max_h=canvas_height)
            
            # Limpa canvas e exibe imagem
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                   image=tk_image, anchor="center")
            
            # Mantém referência da imagem
            self.canvas.image = tk_image
            
        except Exception as e:
            print(f"Erro ao exibir imagem: {e}")
            
    def mark_as_ok(self):
        """Marca a amostra atual como OK."""
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({
                'roi': self.current_roi.copy(),
                'label': 'OK',
                'timestamp': timestamp
            })
            
            # Salva a amostra em disco
            if self.samples_dir:
                try:
                    # Cria diretório se não existir
                    ok_dir = os.path.join(self.samples_dir, "ok")
                    os.makedirs(ok_dir, exist_ok=True)
                    
                    # Formata o timestamp para o nome do arquivo
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"ok_sample_{timestamp_str}.png"
                    file_path = os.path.join(ok_dir, filename)
                    
                    # Salva a imagem
                    cv2.imwrite(file_path, self.current_roi.copy())
                    print(f"Amostra OK salva em: {file_path}")
                except Exception as e:
                    print(f"Erro ao salvar amostra OK: {e}")
            
            # Adiciona ao histórico visual
            self.add_sample_to_history(self.current_roi.copy(), "OK", timestamp)
            
            self.update_info_label()
            self.update_tab_titles()
            self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como OK!")
            
    def mark_as_ng(self):
        """Marca a amostra atual como NG."""
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({
                'roi': self.current_roi.copy(),
                'label': 'NG',
                'timestamp': timestamp
            })
            
            # Salva a amostra em disco
            if self.samples_dir:
                try:
                    # Cria diretório se não existir
                    ng_dir = os.path.join(self.samples_dir, "ng")
                    os.makedirs(ng_dir, exist_ok=True)
                    
                    # Formata o timestamp para o nome do arquivo
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"ng_sample_{timestamp_str}.png"
                    file_path = os.path.join(ng_dir, filename)
                    
                    # Salva a imagem
                    cv2.imwrite(file_path, self.current_roi.copy())
                    print(f"Amostra NG salva em: {file_path}")
                except Exception as e:
                    print(f"Erro ao salvar amostra NG: {e}")
            
            # Adiciona ao histórico visual
            self.add_sample_to_history(self.current_roi.copy(), "NG", timestamp)
            
            self.update_info_label()
            self.update_tab_titles()
            self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como NG!")
            
    def reset_capture_state(self):
        """Reseta o estado de captura."""
        self.current_image = None
        self.current_roi = None
        self.btn_mark_ok.config(state=DISABLED)
        self.btn_mark_ng.config(state=DISABLED)
        self.canvas.delete("all")
        
    def add_sample_to_history(self, roi_image, label, timestamp):
        """Adiciona uma amostra ao histórico visual."""
        try:
            # Redimensiona a imagem para miniatura (100x100)
            thumbnail_size = (100, 100)
            roi_resized = cv2.resize(roi_image, thumbnail_size)
            
            # Converte para formato Tkinter
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)
            roi_tk = ImageTk.PhotoImage(roi_pil)
            
            # Seleciona o frame correto
            if label == "OK":
                parent_frame = self.ok_scrollable_frame
                bg_color = "#e8f5e8"
            else:
                parent_frame = self.ng_scrollable_frame
                bg_color = "#f5e8e8"
            
            # Cria frame para a amostra
            sample_frame = ttk.Frame(parent_frame)
            sample_frame.pack(fill=X, padx=5, pady=2)
            
            # Frame interno com borda colorida
            inner_frame = ttk.Frame(sample_frame, relief="solid", borderwidth=1)
            inner_frame.pack(fill=X, padx=2, pady=2)
            
            # Frame para imagem e informações
            content_frame = ttk.Frame(inner_frame)
            content_frame.pack(fill=X, padx=5, pady=5)
            
            # Label para a imagem
            img_label = ttk.Label(content_frame, image=roi_tk)
            img_label.image = roi_tk  # Mantém referência
            img_label.pack(side=LEFT, padx=(0, 10))
            
            # Frame para informações
            info_frame = ttk.Frame(content_frame)
            info_frame.pack(side=LEFT, fill=BOTH, expand=True)
            
            # Informações da amostra
            time_str = timestamp.strftime("%H:%M:%S")
            date_str = timestamp.strftime("%d/%m/%Y")
            
            ttk.Label(info_frame, text=f"🕒 {time_str}", font=("Arial", 9)).pack(anchor="w")
            ttk.Label(info_frame, text=f"📅 {date_str}", font=("Arial", 8)).pack(anchor="w")
            ttk.Label(info_frame, text=f"📏 {roi_image.shape[1]}x{roi_image.shape[0]}", 
                     font=("Arial", 8)).pack(anchor="w")
            
            # Botão para remover amostra
            remove_btn = ttk.Button(info_frame, text="🗑️", width=3,
                                   command=lambda: self.remove_sample_from_history(sample_frame, label, timestamp))
            remove_btn.pack(anchor="e", pady=(5, 0))
            
        except Exception as e:
            print(f"Erro ao adicionar amostra ao histórico: {e}")
    
    def remove_sample_from_history(self, sample_frame, label, timestamp):
        """Remove uma amostra do histórico visual e da lista."""
        try:
            # Remove da lista de amostras
            self.training_samples = [s for s in self.training_samples 
                                   if not (s['label'] == label and s['timestamp'] == timestamp)]
            
            # Remove o arquivo de amostra do disco
            if self.samples_dir:
                try:
                    # Determina o diretório correto (ok ou ng)
                    sample_dir = os.path.join(self.samples_dir, "ok" if label == "OK" else "ng")
                    if os.path.exists(sample_dir):
                        # Formata o timestamp para o nome do arquivo
                        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                        filename = f"{label.lower()}_sample_{timestamp_str}.png"
                        file_path = os.path.join(sample_dir, filename)
                        
                        # Remove o arquivo se existir
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Arquivo de amostra removido: {file_path}")
                except Exception as e:
                    print(f"Erro ao remover arquivo de amostra: {e}")
            
            # Remove o frame visual
            sample_frame.destroy()
            
            # Atualiza contadores
            self.update_info_label()
            self.update_tab_titles()
            
        except Exception as e:
            print(f"Erro ao remover amostra: {e}")
    
    def update_tab_titles(self):
        """Atualiza os títulos das abas com o número de amostras."""
        ok_count = sum(1 for sample in self.training_samples if sample['label'] == 'OK')
        ng_count = sum(1 for sample in self.training_samples if sample['label'] == 'NG')
        
        self.history_notebook.tab(0, text=f"✅ Amostras OK ({ok_count})")
        self.history_notebook.tab(1, text=f"❌ Amostras NG ({ng_count})")
    
    def clear_training_history(self):
        """Limpa todo o histórico de treinamento."""
        if messagebox.askyesno("Confirmar", "Deseja realmente limpar todo o histórico de treinamento?"):
            # Limpa a lista de amostras
            self.training_samples.clear()
            
            # Limpa os frames visuais
            for widget in self.ok_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in self.ng_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Remove arquivos de amostra do disco
            if self.samples_dir:
                try:
                    # Remove amostras OK
                    ok_dir = os.path.join(self.samples_dir, "ok")
                    if os.path.exists(ok_dir):
                        for filename in os.listdir(ok_dir):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                os.remove(os.path.join(ok_dir, filename))
                    
                    # Remove amostras NG
                    ng_dir = os.path.join(self.samples_dir, "ng")
                    if os.path.exists(ng_dir):
                        for filename in os.listdir(ng_dir):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                os.remove(os.path.join(ng_dir, filename))
                                
                    print("Arquivos de amostra removidos do disco")
                except Exception as e:
                    print(f"Erro ao remover arquivos de amostra: {e}")
            
            # Atualiza interface
            self.update_info_label()
            self.update_tab_titles()
            
            messagebox.showinfo("Sucesso", "Histórico de treinamento limpo!")
    
    def load_existing_samples(self):
        """Carrega amostras existentes do diretório de treinamento."""
        try:
            # Verifica se o diretório de amostras foi definido
            if not self.samples_dir:
                print("Diretório de amostras não definido. Pulando carregamento de amostras existentes.")
                return
                
            # Verifica se existem amostras OK
            ok_samples_dir = os.path.join(self.samples_dir, "ok")
            if os.path.exists(ok_samples_dir):
                for filename in sorted(os.listdir(ok_samples_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_path = os.path.join(ok_samples_dir, filename)
                        try:
                            # Carrega a imagem
                            roi_image = cv2.imread(sample_path)
                            if roi_image is not None:
                                # Extrai timestamp do nome do arquivo
                                timestamp_str = filename.split('_')[2:4]  # ok_sample_YYYYMMDD_HHMMSS
                                if len(timestamp_str) >= 2:
                                    date_part = timestamp_str[0]
                                    time_part = timestamp_str[1].split('.')[0]  # Remove extensão
                                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                                else:
                                    timestamp = datetime.now()
                                
                                # Adiciona à lista de amostras
                                self.training_samples.append({
                                    'roi': roi_image,
                                    'label': 'OK',
                                    'timestamp': timestamp
                                })
                                
                                # Adiciona ao histórico visual
                                self.add_sample_to_history(roi_image, "OK", timestamp)
                        except Exception as e:
                            print(f"Erro ao carregar amostra OK {filename}: {e}")
            
            # Verifica se existem amostras NG
            ng_samples_dir = os.path.join(self.samples_dir, "ng")
            if os.path.exists(ng_samples_dir):
                for filename in sorted(os.listdir(ng_samples_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_path = os.path.join(ng_samples_dir, filename)
                        try:
                            # Carrega a imagem
                            roi_image = cv2.imread(sample_path)
                            if roi_image is not None:
                                # Extrai timestamp do nome do arquivo
                                timestamp_str = filename.split('_')[2:4]  # ng_sample_YYYYMMDD_HHMMSS
                                if len(timestamp_str) >= 2:
                                    date_part = timestamp_str[0]
                                    time_part = timestamp_str[1].split('.')[0]  # Remove extensão
                                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                                else:
                                    timestamp = datetime.now()
                                
                                # Adiciona à lista de amostras
                                self.training_samples.append({
                                    'roi': roi_image,
                                    'label': 'NG',
                                    'timestamp': timestamp
                                })
                                
                                # Adiciona ao histórico visual
                                self.add_sample_to_history(roi_image, "NG", timestamp)
                        except Exception as e:
                            print(f"Erro ao carregar amostra NG {filename}: {e}")
            
            # Atualiza interface
            self.update_info_label()
            self.update_tab_titles()
            
        except Exception as e:
            print(f"Erro ao carregar amostras existentes: {e}")
    
    def update_info_label(self):
        """Atualiza o label de informações."""
        ok_count = sum(1 for sample in self.training_samples if sample['label'] == 'OK')
        ng_count = sum(1 for sample in self.training_samples if sample['label'] == 'NG')
        
        self.info_label.config(text=f"Amostras coletadas: {ok_count} OK, {ng_count} NG")
        
        # Habilita botão de aplicar se há amostras suficientes
        if len(self.training_samples) >= 2:  # Pelo menos 2 amostras
            self.btn_apply_training.config(state=NORMAL)
            # Habilita botões ML se método ML estiver selecionado
            if self.use_ml:
                self.btn_train_ml.config(state=NORMAL)
    
    def on_method_change(self):
        """Callback quando o método de treinamento é alterado."""
        self.use_ml = self.training_method_var.get() == "ml"
        
        if self.use_ml:
            # Mostra botões ML
            self.btn_train_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            self.btn_save_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            # Atualiza texto do botão principal
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento (Tradicional)")
        else:
            # Oculta botões ML
            self.btn_train_ml.pack_forget()
            self.btn_save_ml.pack_forget()
            # Restaura texto do botão principal
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento")
        
        # Atualiza estado dos botões
        self.update_info_label()
    
    def train_ml_model(self):
        """Treina o modelo de machine learning com as amostras coletadas."""
        try:
            if len(self.training_samples) < 4:
                messagebox.showwarning("Aviso", "São necessárias pelo menos 4 amostras (2 OK + 2 NG) para treinamento de ML.")
                return
            elif len(self.training_samples) < 10:
                messagebox.showinfo("Informação", f"Treinando com {len(self.training_samples)} amostras.\nPara melhor precisão, recomenda-se 10+ amostras.")
            
            # Verifica se há amostras OK e NG
            ok_samples = [s for s in self.training_samples if s['label'] == 'OK']
            ng_samples = [s for s in self.training_samples if s['label'] == 'NG']
            
            if not ok_samples or not ng_samples:
                messagebox.showwarning("Aviso", "É necessário ter amostras tanto OK quanto NG para treinamento de ML.")
                return
            
            # Mostra progresso
            progress_window = Toplevel(self)
            progress_window.title("Treinando Modelo ML")
            progress_window.geometry("400x150")
            progress_window.transient(self)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Treinando modelo de machine learning...")
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill=X)
            progress_bar.start()
            
            # Força atualização da interface
            progress_window.update()
            
            # Treina o modelo
            metrics = self.ml_classifier.train(self.training_samples)
            
            # Para a barra de progresso
            progress_bar.stop()
            progress_window.destroy()
            
            # Mostra resultados
            accuracy = metrics.get('accuracy', 0)
            cv_mean = metrics.get('cv_mean', 0)
            cv_std = metrics.get('cv_std', 0)
            n_samples = metrics.get('n_samples', 0)
            
            result_msg = (
                f"🤖 Modelo ML treinado com sucesso!\n\n"
                f"📊 Métricas de Performance:\n"
                f"• Acurácia: {accuracy:.1%}\n"
                f"• Validação Cruzada: {cv_mean:.1%} (±{cv_std:.1%})\n"
                f"• Amostras utilizadas: {n_samples}\n"
                f"• Amostras OK: {metrics.get('n_ok', 0)}\n"
                f"• Amostras NG: {metrics.get('n_ng', 0)}\n\n"
                f"O modelo está pronto para uso!"
            )
            
            messagebox.showinfo("Sucesso", result_msg)
            
            # Habilita botão de salvar
            self.btn_save_ml.config(state=NORMAL)
            
            # Atualiza slot com flag ML
            self.slot_data['use_ml'] = True
            self.slot_data['ml_trained'] = True
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar modelo ML: {str(e)}")
    
    def save_ml_model(self):
        """Salva o modelo ML treinado."""
        try:
            if not self.ml_classifier.is_trained:
                messagebox.showwarning("Aviso", "Nenhum modelo ML foi treinado ainda.")
                return
            
            # Define caminho para salvar o modelo
            template_path = self.slot_data.get('template_path')
            if template_path:
                model_dir = os.path.dirname(template_path)
            else:
                model_dir = get_template_dir()
            
            model_filename = f"ml_model_slot_{self.slot_data['id']}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            
            # Salva o modelo
            if self.ml_classifier.save_model(model_path):
                # Atualiza dados do slot
                self.slot_data['ml_model_path'] = model_path
                self.slot_data['use_ml'] = True
                
                # Salva no banco de dados se possível
                try:
                    if hasattr(self.montagem_instance, 'db_manager') and self.montagem_instance.db_manager:
                        # Atualiza slot no banco
                        self.montagem_instance.db_manager.update_slot(
                            self.slot_data['id'],
                            self.montagem_instance.current_model_id,
                            **self.slot_data
                        )
                except Exception as db_error:
                    print(f"Aviso: Não foi possível salvar no banco de dados: {db_error}")
                
                messagebox.showinfo("Sucesso", f"Modelo ML salvo com sucesso!\n\nCaminho: {model_path}")
            else:
                messagebox.showerror("Erro", "Falha ao salvar o modelo ML.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar modelo ML: {str(e)}")
            
    def apply_training(self):
        """Aplica o treinamento coletado para melhorar a precisão do slot."""
        try:
            if len(self.training_samples) < 2:
                messagebox.showwarning("Aviso", "São necessárias pelo menos 2 amostras para treinamento.")
                return
                
            # Analisa as amostras para ajustar parâmetros
            ok_samples = [s['roi'] for s in self.training_samples if s['label'] == 'OK']
            ng_samples = [s['roi'] for s in self.training_samples if s['label'] == 'NG']
            
            if not ok_samples:
                messagebox.showwarning("Aviso", "É necessária pelo menos uma amostra OK.")
                return
            
            if self.use_ml:
                # Modo Machine Learning
                if not self.ml_classifier.is_trained:
                    messagebox.showwarning("Aviso", "Modelo ML não foi treinado ainda. Treine o modelo primeiro.")
                    return
                
                # Testa o modelo com as amostras atuais
                correct_predictions = 0
                total_predictions = 0
                
                for sample in self.training_samples:
                    prediction = self.ml_classifier.predict(sample['roi'])
                    expected = 1 if sample['label'] == 'OK' else 0
                    if prediction == expected:
                        correct_predictions += 1
                    total_predictions += 1
                
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Atualiza configurações do slot para usar ML
                self.slot_data['use_ml'] = True
                self.slot_data['ml_trained'] = True
                self.slot_data['ml_accuracy'] = accuracy
                
                # Salva template melhorado se há amostras OK
                if ok_samples:
                    self.update_template_with_best_sample(ok_samples)
                
                # Atualiza o slot na instância principal
                self.montagem_instance.update_slot_data(self.slot_data)
                
                # Marca modelo como modificado
                self.montagem_instance.mark_model_modified()
                
                result_msg = (
                    f"🤖 Treinamento ML aplicado!\n\n"
                    f"📊 Acurácia nas amostras: {accuracy:.1%}\n"
                    f"✅ Amostras utilizadas: {len(self.training_samples)}\n\n"
                    f"O slot agora usará Machine Learning para classificação!"
                )
                
                messagebox.showinfo("Sucesso", result_msg)
                self.destroy()
                
            else:
                # Modo tradicional (threshold)
                new_threshold = self.calculate_optimal_threshold(ok_samples, ng_samples)
                
                if new_threshold is not None:
                    # Atualiza o slot com o novo limiar
                    old_threshold = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 0.8))
                    self.slot_data['correlation_threshold'] = new_threshold
                    self.slot_data['use_ml'] = False
                    
                    # Salva um template melhorado se há amostras OK
                    if ok_samples:
                        self.update_template_with_best_sample(ok_samples)
                    
                    # Atualiza o slot na instância principal
                    self.montagem_instance.update_slot_data(self.slot_data)
                    
                    # Marca modelo como modificado
                    self.montagem_instance.mark_model_modified()
                    
                    messagebox.showinfo("Sucesso", 
                        f"Treinamento aplicado!\n\n"
                        f"Limiar anterior: {old_threshold:.3f}\n"
                        f"Novo limiar: {new_threshold:.3f}\n\n"
                        f"Amostras utilizadas: {len(self.training_samples)}")
                    
                    self.destroy()
                else:
                    messagebox.showerror("Erro", "Não foi possível calcular novo limiar.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao aplicar treinamento: {str(e)}")
            
    def calculate_optimal_threshold(self, ok_samples, ng_samples):
        """Calcula o limiar ótimo baseado nas amostras de treinamento."""
        try:
            # Validações iniciais
            if not ok_samples:
                print("Erro: Nenhuma amostra OK fornecida")
                return None
                
            # Carrega template atual ou cria um temporário
            template_path = self.slot_data.get('template_path')
            template = None
            
            if template_path and Path(template_path).exists():
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                print(f"Template carregado de: {template_path}")
            
            # Se não há template ou falhou ao carregar, usa a primeira amostra OK como template
            if template is None:
                if not ok_samples:
                    print("Erro: Não há template nem amostras OK para usar como referência")
                    return None
                    
                print("Template não encontrado. Usando primeira amostra OK como referência.")
                first_sample = ok_samples[0]
                template = cv2.cvtColor(first_sample, cv2.COLOR_BGR2GRAY)
                
                # Salva template temporário se possível
                if template_path:
                    try:
                        # Cria diretório se não existir
                        Path(template_path).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(template_path, template)
                        print(f"Template temporário salvo em: {template_path}")
                    except Exception as e:
                        print(f"Aviso: Não foi possível salvar template temporário: {e}")
                
            # Calcula correlações para amostras OK
            ok_correlations = []
            for i, roi in enumerate(ok_samples):
                try:
                    if roi is None or roi.size == 0:
                        print(f"Aviso: Amostra OK {i} é inválida")
                        continue
                        
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    if roi_gray.size == 0:
                        print(f"Aviso: Amostra OK {i} resultou em imagem vazia")
                        continue
                    
                    # Redimensiona template se necessário
                    if roi_gray.shape != template.shape:
                        template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                    else:
                        template_resized = template
                        
                    # Template matching
                    result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Valida o resultado
                    if not np.isnan(max_val) and not np.isinf(max_val):
                        ok_correlations.append(max_val)
                    else:
                        print(f"Aviso: Correlação inválida para amostra OK {i}: {max_val}")
                        
                except Exception as e:
                    print(f"Erro ao processar amostra OK {i}: {e}")
                    continue
                
            # Calcula correlações para amostras NG (se existirem)
            ng_correlations = []
            for i, roi in enumerate(ng_samples):
                try:
                    if roi is None or roi.size == 0:
                        print(f"Aviso: Amostra NG {i} é inválida")
                        continue
                        
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    if roi_gray.size == 0:
                        print(f"Aviso: Amostra NG {i} resultou em imagem vazia")
                        continue
                    
                    if roi_gray.shape != template.shape:
                        template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                    else:
                        template_resized = template
                        
                    result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Valida o resultado
                    if not np.isnan(max_val) and not np.isinf(max_val):
                        ng_correlations.append(max_val)
                    else:
                        print(f"Aviso: Correlação inválida para amostra NG {i}: {max_val}")
                        
                except Exception as e:
                    print(f"Erro ao processar amostra NG {i}: {e}")
                    continue
                
            # Verifica se temos correlações válidas
            if not ok_correlations:
                print("Erro: Nenhuma correlação válida foi calculada para amostras OK")
                return None
                
            print(f"Correlações OK calculadas: {len(ok_correlations)} amostras")
            print(f"Correlações NG calculadas: {len(ng_correlations)} amostras")
            
            # Calcula limiar ótimo
            min_ok = min(ok_correlations)
            max_ok = max(ok_correlations)
            avg_ok = sum(ok_correlations) / len(ok_correlations)
            
            print(f"Estatísticas OK - Min: {min_ok:.3f}, Max: {max_ok:.3f}, Média: {avg_ok:.3f}")
            
            if ng_correlations:
                min_ng = min(ng_correlations)
                max_ng = max(ng_correlations)
                avg_ng = sum(ng_correlations) / len(ng_correlations)
                
                print(f"Estatísticas NG - Min: {min_ng:.3f}, Max: {max_ng:.3f}, Média: {avg_ng:.3f}")
                
                # Verifica se há separação clara entre OK e NG
                if min_ok > max_ng:
                    # Caso ideal: há separação clara
                    new_threshold = (min_ok + max_ng) / 2
                    print(f"Separação clara detectada. Limiar calculado: {new_threshold:.3f}")
                else:
                    # Caso problemático: sobreposição entre OK e NG
                    # Usa a média das amostras OK menos uma margem de segurança
                    new_threshold = avg_ok * 0.85
                    print(f"Sobreposição detectada. Usando limiar conservador: {new_threshold:.3f}")
                
                # Garante que está dentro de limites razoáveis
                new_threshold = max(0.3, min(0.95, new_threshold))
            else:
                # Se não há amostras NG, usa um valor conservador baseado na média OK
                new_threshold = max(0.5, min(0.9, avg_ok * 0.9))
                print(f"Apenas amostras OK. Limiar conservador: {new_threshold:.3f}")
                
            # Validação final
            if np.isnan(new_threshold) or np.isinf(new_threshold):
                print(f"Erro: Limiar calculado é inválido: {new_threshold}")
                return None
                
            print(f"Limiar final calculado: {new_threshold:.3f}")
            return new_threshold
                
            return None
            
        except Exception as e:
            print(f"Erro ao calcular limiar: {e}")
            return None
            
    def update_template_with_best_sample(self, ok_samples):
        """Atualiza o template com a melhor amostra OK."""
        try:
            template_path = self.slot_data.get('template_path')
            if not template_path:
                return
                
            # Encontra a melhor amostra (maior correlação com template atual)
            current_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if current_template is None:
                return
                
            best_sample = None
            best_correlation = -1
            
            for roi in ok_samples:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Redimensiona para comparar
                if roi_gray.shape != current_template.shape:
                    roi_resized = cv2.resize(roi_gray, (current_template.shape[1], current_template.shape[0]))
                else:
                    roi_resized = roi_gray
                    
                # Calcula correlação
                result = cv2.matchTemplate(roi_resized, current_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_correlation:
                    best_correlation = max_val
                    best_sample = roi_gray
                    
            # Salva o melhor template
            if best_sample is not None:
                # Redimensiona para o tamanho original do template
                if best_sample.shape != current_template.shape:
                    best_sample = cv2.resize(best_sample, (current_template.shape[1], current_template.shape[0]))
                    
                cv2.imwrite(template_path, best_sample)
                print(f"Template atualizado com melhor amostra (correlação: {best_correlation:.3f})")
                
        except Exception as e:
            print(f"Erro ao atualizar template: {e}")
            
    def cancel(self):
        """Cancela o treinamento."""
        self.destroy()


class SystemConfigDialog(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("⚙️ Configurações do Sistema")
        self.geometry("550x750")  # Aumentado para acomodar todas as configurações
        self.resizable(True, True)  # Permitir redimensionamento para melhor visualização
        self.transient(parent)
        self.grab_set()
        
        # Importa o módulo colorchooser para seleção de cores
        
        # Carrega as configurações de estilo atuais
        self.style_config = load_style_config()
        
        self.result = False
        self.center_window()
        self.setup_ui()
    
    def center_window(self):
        self.update_idletasks()
        width = 550
        height = 750
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_ui(self):
        # Criar um canvas com scrollbar para acomodar todas as configurações
        container = ttk.Frame(self)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Criar canvas com tamanho adequado
        canvas = Canvas(container, width=530, height=700)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, width=520)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Adicionar evento de rolagem com o mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configurar o layout
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame principal dentro do scrollable_frame
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Configurações ORB
        orb_frame = ttk.LabelFrame(main_frame, text="Configurações ORB (Alinhamento de Imagem)")
        orb_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(orb_frame, text="Número de Features:").pack(anchor="w", padx=5, pady=2)
        self.orb_features_var = ttk.IntVar(value=ORB_FEATURES)
        features_frame = ttk.Frame(orb_frame)
        features_frame.pack(fill=X, padx=5, pady=5)
        
        self.features_scale = ttk.Scale(features_frame, from_=1000, to=10000, variable=self.orb_features_var, orient=HORIZONTAL)
        self.features_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.features_label = ttk.Label(features_frame, text=f"{self.orb_features_var.get()}", width=8)
        self.features_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_features_label(val):
            self.features_label.config(text=f"{int(float(val))}")
        self.features_scale.config(command=update_features_label)
        
        ttk.Label(orb_frame, text="Fator de Escala:").pack(anchor="w", padx=5, pady=(10, 2))
        self.scale_factor_var = ttk.DoubleVar(value=ORB_SCALE_FACTOR)
        scale_frame = ttk.Frame(orb_frame)
        scale_frame.pack(fill=X, padx=5, pady=5)
        
        self.scale_scale = ttk.Scale(scale_frame, from_=1.1, to=2.0, variable=self.scale_factor_var, orient=HORIZONTAL)
        self.scale_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.scale_label = ttk.Label(scale_frame, text=f"{self.scale_factor_var.get():.2f}", width=8)
        self.scale_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_scale_label(val):
            self.scale_label.config(text=f"{float(val):.2f}")
        self.scale_scale.config(command=update_scale_label)
        
        ttk.Label(orb_frame, text="Número de Níveis:").pack(anchor="w", padx=5, pady=(10, 2))
        self.n_levels_var = ttk.IntVar(value=ORB_N_LEVELS)
        levels_spin = ttk.Spinbox(orb_frame, from_=4, to=16, textvariable=self.n_levels_var, width=10)
        levels_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações de Canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Configurações de Visualização")
        canvas_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(canvas_frame, text="Largura Máxima do Preview:").pack(anchor="w", padx=5, pady=2)
        self.preview_w_var = ttk.IntVar(value=PREVIEW_W)
        w_spin = ttk.Spinbox(canvas_frame, from_=400, to=1600, increment=100, textvariable=self.preview_w_var, width=10)
        w_spin.pack(anchor="w", padx=5, pady=5)
        
        ttk.Label(canvas_frame, text="Altura Máxima do Preview:").pack(anchor="w", padx=5, pady=(10, 2))
        self.preview_h_var = ttk.IntVar(value=PREVIEW_H)
        h_spin = ttk.Spinbox(canvas_frame, from_=300, to=1200, increment=100, textvariable=self.preview_h_var, width=10)
        h_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações Padrão de Detecção
        detection_frame = ttk.LabelFrame(main_frame, text="Configurações Padrão de Detecção")
        detection_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(detection_frame, text="Limiar de Correlação Padrão (Clips):").pack(anchor="w", padx=5, pady=2)
        self.thr_corr_var = ttk.DoubleVar(value=THR_CORR)
        corr_frame = ttk.Frame(detection_frame)
        corr_frame.pack(fill=X, padx=5, pady=5)
        
        self.corr_scale = ttk.Scale(corr_frame, from_=0.1, to=1.0, variable=self.thr_corr_var, orient=HORIZONTAL)
        self.corr_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.corr_label = ttk.Label(corr_frame, text=f"{self.thr_corr_var.get():.2f}", width=8)
        self.corr_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_corr_label(val):
            self.corr_label.config(text=f"{float(val):.2f}")
        self.corr_scale.config(command=update_corr_label)
        
        ttk.Label(detection_frame, text="Pixels Mínimos Padrão (Template Matching):").pack(anchor="w", padx=5, pady=(10, 2))
        self.min_px_var = ttk.IntVar(value=MIN_PX)
        px_spin = ttk.Spinbox(detection_frame, from_=1, to=1000, textvariable=self.min_px_var, width=10)
        px_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações de Aparência por Local
        appearance_frame = ttk.LabelFrame(main_frame, text="Configurações de Aparência por Local")
        appearance_frame.pack(fill=X, pady=(0, 10))
        
        # Configurações de Fonte para Diferentes Locais
        font_frame = ttk.Frame(appearance_frame)
        font_frame.pack(fill=X, padx=5, pady=5)
        
        # Fonte para Slots
        ttk.Label(font_frame, text="Tamanho da Fonte para Slots:").pack(anchor="w", padx=5, pady=(10, 2))
        self.slot_font_size_var = ttk.IntVar(value=int(self.style_config.get("slot_font_size", 10)))
        slot_font_spin = ttk.Spinbox(font_frame, from_=8, to=24, textvariable=self.slot_font_size_var, width=10)
        slot_font_spin.pack(anchor="w", padx=5, pady=5)
        
        # Fonte para Resultados
        ttk.Label(font_frame, text="Tamanho da Fonte para Resultados:").pack(anchor="w", padx=5, pady=(10, 2))
        self.result_font_size_var = ttk.IntVar(value=int(self.style_config.get("result_font_size", 10)))
        result_font_spin = ttk.Spinbox(font_frame, from_=8, to=24, textvariable=self.result_font_size_var, width=10)
        result_font_spin.pack(anchor="w", padx=5, pady=5)
        
        # Fonte para Botões
        ttk.Label(font_frame, text="Tamanho da Fonte para Botões:").pack(anchor="w", padx=5, pady=(10, 2))
        self.button_font_size_var = ttk.IntVar(value=int(self.style_config.get("button_font_size", 9)))
        button_font_spin = ttk.Spinbox(font_frame, from_=8, to=20, textvariable=self.button_font_size_var, width=10)
        button_font_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações de HUD e Inspeção
        hud_frame = ttk.LabelFrame(main_frame, text="Configurações de HUD e Inspeção")
        hud_frame.pack(fill=X, pady=(0, 10))
        
        # Configurações de HUD
        hud_config_frame = ttk.Frame(hud_frame)
        hud_config_frame.pack(fill=X, padx=5, pady=5)
        
        # Tamanho da Fonte do HUD
        ttk.Label(hud_config_frame, text="Tamanho da Fonte do HUD:").pack(anchor="w", padx=5, pady=(10, 2))
        self.hud_font_size_var = ttk.IntVar(value=int(self.style_config.get("hud_font_size", 12)))
        hud_font_spin = ttk.Spinbox(hud_config_frame, from_=8, to=28, textvariable=self.hud_font_size_var, width=10)
        hud_font_spin.pack(anchor="w", padx=5, pady=5)
        
        # Opacidade do HUD
        ttk.Label(hud_config_frame, text="Opacidade do HUD (%):").pack(anchor="w", padx=5, pady=(10, 2))
        self.hud_opacity_var = ttk.IntVar(value=int(self.style_config.get("hud_opacity", 80)))
        opacity_frame = ttk.Frame(hud_config_frame)
        opacity_frame.pack(fill=X, padx=5, pady=5)
        
        self.opacity_scale = ttk.Scale(opacity_frame, from_=10, to=100, variable=self.hud_opacity_var, orient=HORIZONTAL)
        self.opacity_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.opacity_label = ttk.Label(opacity_frame, text=f"{self.hud_opacity_var.get()}%", width=8)
        self.opacity_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_opacity_label(val):
            self.opacity_label.config(text=f"{int(float(val))}%")
        self.opacity_scale.config(command=update_opacity_label)
        
        # Posição do HUD
        ttk.Label(hud_config_frame, text="Posição do HUD:").pack(anchor="w", padx=5, pady=(10, 2))
        self.hud_position_var = ttk.StringVar(value=self.style_config.get("hud_position", "top-right"))
        position_frame = ttk.Frame(hud_config_frame)
        position_frame.pack(fill=X, padx=5, pady=5)
        
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        position_combo = ttk.Combobox(position_frame, textvariable=self.hud_position_var, values=positions, state="readonly", width=15)
        position_combo.pack(side=LEFT, padx=5)
        
        # Mostrar informações adicionais no HUD
        self.show_fps_var = ttk.BooleanVar(value=self.style_config.get("show_fps", True))
        show_fps_check = ttk.Checkbutton(hud_config_frame, text="Mostrar FPS", variable=self.show_fps_var)
        show_fps_check.pack(anchor="w", padx=5, pady=5)
        
        self.show_timestamp_var = ttk.BooleanVar(value=self.style_config.get("show_timestamp", True))
        show_timestamp_check = ttk.Checkbutton(hud_config_frame, text="Mostrar Timestamp", variable=self.show_timestamp_var)
        show_timestamp_check.pack(anchor="w", padx=5, pady=5)
        
        # Cores
        colors_frame = ttk.Frame(appearance_frame)
        colors_frame.pack(fill=X, padx=5, pady=5)
        
        # Cor de Fundo
        bg_color_frame = ttk.Frame(colors_frame)
        bg_color_frame.pack(fill=X, pady=2)
        
        ttk.Label(bg_color_frame, text="Cor de Fundo:").pack(side=LEFT, padx=5)
        self.bg_color_var = ttk.StringVar(value=self.style_config.get("background_color", "#1E1E1E"))
        bg_color_entry = ttk.Entry(bg_color_frame, textvariable=self.bg_color_var, width=10)
        bg_color_entry.pack(side=LEFT, padx=5)
        
        # Botão para escolher cor
        def choose_bg_color():
            color = colorchooser.askcolor(initialcolor=self.bg_color_var.get(), title="Escolher Cor de Fundo")
            if color and color[1]:
                self.bg_color_var.set(color[1])
        
        ttk.Button(bg_color_frame, text="Escolher", command=choose_bg_color).pack(side=LEFT, padx=5)
        
        # Cor do Texto
        text_color_frame = ttk.Frame(colors_frame)
        text_color_frame.pack(fill=X, pady=2)
        
        ttk.Label(text_color_frame, text="Cor do Texto:").pack(side=LEFT, padx=5)
        self.text_color_var = ttk.StringVar(value=self.style_config.get("text_color", "#FFFFFF"))
        text_color_entry = ttk.Entry(text_color_frame, textvariable=self.text_color_var, width=10)
        text_color_entry.pack(side=LEFT, padx=5)
        
        # Botão para escolher cor
        def choose_text_color():
            color = colorchooser.askcolor(initialcolor=self.text_color_var.get(), title="Escolher Cor do Texto")
            if color and color[1]:
                self.text_color_var.set(color[1])
        
        ttk.Button(text_color_frame, text="Escolher", command=choose_text_color).pack(side=LEFT, padx=5)
        
        # Cor OK
        ok_color_frame = ttk.Frame(colors_frame)
        ok_color_frame.pack(fill=X, pady=2)
        
        ttk.Label(ok_color_frame, text="Cor OK:").pack(side=LEFT, padx=5)
        self.ok_color_var = ttk.StringVar(value=self.style_config.get("ok_color", "#8cde81"))
        ok_color_entry = ttk.Entry(ok_color_frame, textvariable=self.ok_color_var, width=10)
        ok_color_entry.pack(side=LEFT, padx=5)
        
        # Botão para escolher cor
        def choose_ok_color():
            color = colorchooser.askcolor(initialcolor=self.ok_color_var.get(), title="Escolher Cor OK")
            if color and color[1]:
                self.ok_color_var.set(color[1])
        
        ttk.Button(ok_color_frame, text="Escolher", command=choose_ok_color).pack(side=LEFT, padx=5)
        
        # Cor NG
        ng_color_frame = ttk.Frame(colors_frame)
        ng_color_frame.pack(fill=X, pady=2)
        
        ttk.Label(ng_color_frame, text="Cor NG:").pack(side=LEFT, padx=5)
        self.ng_color_var = ttk.StringVar(value=self.style_config.get("ng_color", "#e7472c"))
        ng_color_entry = ttk.Entry(ng_color_frame, textvariable=self.ng_color_var, width=10)
        ng_color_entry.pack(side=LEFT, padx=5)
        
        # Botão para escolher cor
        def choose_ng_color():
            color = colorchooser.askcolor(initialcolor=self.ng_color_var.get(), title="Escolher Cor NG")
            if color and color[1]:
                self.ng_color_var.set(color[1])
        
        ttk.Button(ng_color_frame, text="Escolher", command=choose_ng_color).pack(side=LEFT, padx=5)
        
        # Botões - usando um frame com espaçamento melhor
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(20, 10), padx=10)
        
        # Distribuir os botões uniformemente
        save_btn = ttk.Button(button_frame, text="Salvar", command=self.save_config)
        save_btn.pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)
        
        restore_btn = ttk.Button(button_frame, text="Restaurar Padrões", command=self.restore_defaults)
        restore_btn.pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)
        
        cancel_btn = ttk.Button(button_frame, text="Cancelar", command=self.cancel)
        cancel_btn.pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)
    
    def save_config(self):
        """Salva as configurações do sistema"""
        global ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX, orb
        
        try:
            # Atualiza variáveis globais
            ORB_FEATURES = int(self.orb_features_var.get())
            ORB_SCALE_FACTOR = float(self.scale_factor_var.get())
            ORB_N_LEVELS = int(self.n_levels_var.get())
            PREVIEW_W = int(self.preview_w_var.get())
            PREVIEW_H = int(self.preview_h_var.get())
            THR_CORR = float(self.thr_corr_var.get())
            MIN_PX = int(self.min_px_var.get())
            
            # Reinicializa detector ORB com novos parâmetros
            try:
                orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_N_LEVELS)
                print(f"Detector ORB reinicializado: features={ORB_FEATURES}, scale={ORB_SCALE_FACTOR}, levels={ORB_N_LEVELS}")
            except Exception as e:
                print(f"Erro ao reinicializar ORB: {e}")
                messagebox.showwarning("Aviso", "Erro ao reinicializar detector ORB. O alinhamento pode não funcionar.")
            
            # Salvar configurações de estilo
            style_config = {}
            style_config["slot_font_size"] = self.slot_font_size_var.get()
            style_config["result_font_size"] = self.result_font_size_var.get()
            style_config["button_font_size"] = self.button_font_size_var.get()
            style_config["background_color"] = self.bg_color_var.get()
            style_config["text_color"] = self.text_color_var.get()
            style_config["ok_color"] = self.ok_color_var.get()
            style_config["ng_color"] = self.ng_color_var.get()
            
            # Salvar configurações de HUD e Inspeção
            style_config["hud_font_size"] = self.hud_font_size_var.get()
            style_config["hud_opacity"] = self.hud_opacity_var.get()
            style_config["hud_position"] = self.hud_position_var.get()
            style_config["show_fps"] = self.show_fps_var.get()
            style_config["show_timestamp"] = self.show_timestamp_var.get()
            
            # Salvar no arquivo de configuração de estilo
            save_style_config(style_config)
            
            # Aplicar as configurações de estilo imediatamente
            apply_style_config(style_config)
            
            self.result = True
            messagebox.showinfo("Sucesso", "Configurações salvas com sucesso!")
            # Desvincular o evento de rolagem do mouse antes de fechar
            try:
                self.unbind_all("<MouseWheel>")
            except:
                pass
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {str(e)}")
    
    def restore_defaults(self):
        """Restaura configurações padrão"""
        # Restaura configurações ORB
        self.orb_features_var.set(5000)
        self.scale_factor_var.set(1.2)
        self.n_levels_var.set(8)
        self.preview_w_var.set(800)
        self.preview_h_var.set(600)
        self.thr_corr_var.set(0.1)
        self.min_px_var.set(10)
        
        # Atualiza labels
        self.features_label.config(text="5000")
        self.scale_label.config(text="1.20")
        self.corr_label.config(text="0.10")
        
        # Restaura configurações de estilo
        self.slot_font_size_var.set(10)
        self.result_font_size_var.set(10)
        self.button_font_size_var.set(9)
        self.bg_color_var.set("#1E1E1E")
        self.text_color_var.set("#FFFFFF")
        self.ok_color_var.set("#8cde81")
        self.ng_color_var.set("#e7472c")
    
    def cancel(self):
        """Cancela a edição"""
        # Desvincular o evento de rolagem do mouse antes de fechar
        try:
            self.unbind_all("<MouseWheel>")
        except:
            pass
        self.destroy()


class MontagemWindow(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Inicializa o gerenciador de banco de dados
        # Usa caminho absoluto baseado na raiz do projeto
        db_path = MODEL_DIR / "models.db"
        self.db_manager = DatabaseManager(str(db_path))
        
        # Dados da aplicação
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.slots = []
        self.selected_slot_id = None
        self.current_model_id = None  # ID do modelo atual no banco
        self.current_model = None  # Dados do modelo atual
        self.model_modified = False  # Flag para indicar se o modelo foi modificado
        
        # Estado do desenho
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        
        # Flag para prevenir loop infinito na seleção
        self._selecting_slot = False
        
        # Flag para prevenir múltiplos cliques simultâneos no botão de edição
        self._processing_edit_click = False
        
        # Controle de webcam
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        self.camera = None
        self.live_capture = False
        self.live_view = False
        self.latest_frame = None
        
        # Variáveis de ferramentas de edição
        self.current_drawing_mode = "rectangle"
        self.editing_handle = None
        
        # Configura a interface primeiro
        self.setup_ui()
        self.update_button_states()
        
        # Inicia câmera em segundo plano após inicialização completa
        if self.available_cameras:
            self.after(500, lambda: self.start_background_camera_direct(self.available_cameras[0]))
        
    def start_background_camera_direct(self, camera_index):
        """Inicia a câmera diretamente em segundo plano com índice específico."""
        try:
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução padrão para inicialização rápida
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            print(f"Webcam {camera_index} inicializada com sucesso em segundo plano")
            
            # Inicia captura de frames em thread separada
            self.start_background_frame_capture()
            
        except Exception as e:
            print(f"Erro ao inicializar webcam {camera_index}: {e}")
            self.camera = None
            self.live_capture = False
    

    
    
    def start_background_frame_capture(self):
        """Inicia captura de frames em segundo plano sem exibir no canvas."""
        def capture_loop():
            while self.live_capture and self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.latest_frame = frame.copy()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura em segundo plano: {e}")
                    break
        
        # Inicia thread para captura contínua
        import threading
        self.background_thread = threading.Thread(target=capture_loop, daemon=True)
        self.background_thread.start()
    
    def mark_model_modified(self):
        """Marca o modelo como modificado e atualiza o status."""
        if not self.model_modified:
            self.model_modified = True
            self.update_status_display()
    
    def mark_model_saved(self):
        """Marca o modelo como salvo e atualiza o status."""
        if self.model_modified:
            self.model_modified = False
            self.update_status_display()
    
    def update_status_display(self):
        """Atualiza a exibição do status baseado no estado atual."""
        if self.img_original is None:
            self.status_var.set("Carregue uma imagem para começar")
        elif not self.slots:
            self.status_var.set("Imagem carregada - Desenhe slots para criar o modelo")
        elif self.model_modified:
            self.status_var.set("Modelo modificado - Salve as alterações")
        else:
            model_name = "Modelo atual"
            if self.current_model_id:
                try:
                    modelo = self.db_manager.load_modelo(self.current_model_id)
                    model_name = modelo['nome']
                except:
                    pass
            self.status_var.set(f"Modelo: {model_name} - {len(self.slots)} slots")
    
    def setup_ui(self):
        """Monta o layout principal da janela de montagem."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self._create_left_panel(main_frame)
        self._create_center_panel(main_frame)
        self._create_right_panel(main_frame)
        self._create_status_bar()

        self.show_default_right_panel()

    def _create_left_panel(self, parent):
        """Cria painel esquerdo com controles organizados em abas."""
        left_panel = ttk.Frame(parent)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))

        control_nb = ttk.Notebook(left_panel)
        control_nb.pack(fill=BOTH, expand=True)

        input_tab = ttk.Frame(control_nb)
        control_nb.add(input_tab, text="Entrada")

        edit_tab = ttk.Frame(control_nb)
        control_nb.add(edit_tab, text="Edição")

        help_tab = ttk.Frame(control_nb)
        control_nb.add(help_tab, text="Ajuda")

        # === Seção de Imagem ===
        img_frame = ttk.LabelFrame(input_tab, text="Imagem")
        img_frame.pack(fill=X, pady=(0, 10))
        self.btn_load_image = ttk.Button(img_frame, text="Carregar Imagem",
                                         command=self.load_image)
        self.btn_load_image.pack(fill=X, padx=5, pady=2)

        # === Seção de Webcam ===
        webcam_frame = ttk.LabelFrame(input_tab, text="Webcam")
        webcam_frame.pack(fill=X, pady=(0, 10))
        camera_selection_frame = ttk.Frame(webcam_frame)
        camera_selection_frame.pack(fill=X, padx=5, pady=2)
        ttk.Label(camera_selection_frame, text="Câmera:").pack(side=LEFT)
        self.camera_combo = Combobox(camera_selection_frame,
                                    values=[str(i) for i in self.available_cameras],
                                    state="readonly", width=5)
        self.camera_combo.pack(side=RIGHT)
        if self.available_cameras:
            self.camera_combo.set(str(self.available_cameras[0]))
        self.btn_capture = ttk.Button(webcam_frame, text="Capturar Imagem",
                                      command=self.capture_from_webcam)
        self.btn_capture.pack(fill=X, padx=5, pady=2)

        # === Seção de Modelo ===
        model_frame = ttk.LabelFrame(input_tab, text="Modelo")
        model_frame.pack(fill=X, pady=(0, 10))
        self.btn_load_model = ttk.Button(model_frame, text="Carregar Modelo",
                                         command=self.load_model_dialog)
        self.btn_load_model.pack(fill=X, padx=5, pady=2)
        self.btn_save_model = ttk.Button(model_frame, text="Salvar Modelo",
                                         command=self.save_model)
        self.btn_save_model.pack(fill=X, padx=5, pady=2)

        # === Ferramentas de Edição ===
        tools_frame = ttk.LabelFrame(edit_tab, text="Ferramentas de Edição")
        tools_frame.pack(fill=X, pady=(0, 10))
        mode_frame = ttk.Frame(tools_frame)
        mode_frame.pack(fill=X, padx=5, pady=5)
        ttk.Label(mode_frame, text="Modo de Desenho:").pack(anchor="w")
        self.drawing_mode = StringVar(value="rectangle")
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack(fill=X, pady=2)
        self.style = ttk.Style()
        self.style.configure("TRadiobutton", background="#1E1E1E", foreground="white")
        self.style.map("TRadiobutton",
                       background=[('active', '#1E1E1E'), ('selected', '#1E1E1E')],
                       foreground=[('active', 'white'), ('selected', 'white')])
        self.btn_rect_mode = ttk.Radiobutton(mode_buttons_frame, text="Retângulo",
                                             variable=self.drawing_mode, value="rectangle",
                                             command=self.set_drawing_mode,
                                             style="TRadiobutton")
        self.btn_rect_mode.pack(side=LEFT, padx=(0, 5))
        self.btn_exclusion_mode = ttk.Radiobutton(mode_buttons_frame, text="Exclusão",
                                                  variable=self.drawing_mode, value="exclusion",
                                                  command=self.set_drawing_mode,
                                                  style="TRadiobutton")
        self.btn_exclusion_mode.pack(side=LEFT)
        self.tool_status_var = StringVar(value="Modo: Retângulo")
        ttk.Label(tools_frame, textvariable=self.tool_status_var,
                  font=("Arial", 8), foreground="#666").pack(padx=5, pady=(0, 5))

        # === Slots ===
        slots_frame = ttk.LabelFrame(edit_tab, text="Slots")
        slots_frame.pack(fill=X, pady=(0, 10))
        self.btn_clear_slots = ttk.Button(slots_frame, text="Limpar Todos os Slots",
                                          command=self.clear_slots)
        self.btn_clear_slots.pack(fill=X, padx=5, pady=2)
        self.btn_delete_slot = ttk.Button(slots_frame, text="Deletar Slot Selecionado",
                                          command=self.delete_selected_slot)
        self.btn_delete_slot.pack(fill=X, padx=5, pady=2)
        self.btn_train_slot = ttk.Button(slots_frame, text="Treinar Slot Selecionado",
                                         command=self.train_selected_slot)
        self.btn_train_slot.pack(fill=X, padx=5, pady=2)
        self.slot_info_frame = ttk.LabelFrame(slots_frame, text="Informações do Slot Selecionado")
        self.slot_info_frame.pack(fill=X, padx=5, pady=5)
        self.slot_info_label = ttk.Label(self.slot_info_frame, text="Nenhum slot selecionado",
                                         justify=LEFT)
        self.slot_info_label.pack(fill=X, padx=5, pady=5)

        # === Ajuda ===
        help_frame = ttk.LabelFrame(help_tab, text="Ajuda")
        help_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        self.btn_help = ttk.Button(help_frame, text="Mostrar Ajuda",
                                   command=self.show_help)
        self.btn_help.pack(fill=X, padx=5, pady=5)
        self.btn_config = ttk.Button(help_frame, text="⚙️ Configurações do Sistema",
                                     command=self.open_system_config)
        self.btn_config.pack(fill=X, padx=5, pady=(0, 5))

    def _create_center_panel(self, parent):
        """Cria o painel central com o canvas de edição."""
        center_panel = ttk.Frame(parent)
        center_panel.pack(side=LEFT, fill=BOTH, expand=True)

        canvas_frame = ttk.LabelFrame(center_panel, text="Editor de Malha")
        canvas_frame.pack(fill=BOTH, expand=True)

        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)

        self.canvas = Canvas(canvas_container, bg="#2C3E50",
                             yscrollcommand=v_scrollbar.set,
                             xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_canvas_zoom)
        self.canvas.bind("<Button-2>", self.on_canvas_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_canvas_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_canvas_pan_end)

        self.zoom_level = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0

    def _create_right_panel(self, parent):
        """Cria o painel direito usado para edição de slots."""
        self.right_panel = ttk.Frame(parent, width=350)
        self.right_panel.pack(side=RIGHT, fill=Y, padx=(10, 0), pady=10, expand=False)
        self.right_panel.pack_propagate(False)

    def _create_status_bar(self):
        """Cria a barra de status inferior."""
        self.status_var = StringVar()
        self.status_var.set("Carregue uma imagem para começar")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side=BOTTOM, fill=X, padx=5, pady=2)

    def toggle_live_capture_manual_inspection(self):
        """Alterna o modo de captura contínua com inspeção manual (ativada pelo Enter)."""
        if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            self.stop_live_capture_manual_inspection()
        else:
            self.start_live_capture_manual_inspection()

    def start_live_capture_manual_inspection(self):
        """Inicia captura contínua com inspeção manual."""
        if self.live_capture:
            self.stop_live_capture()
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            import platform
            is_windows = platform.system() == 'Windows'
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.live_capture = True
            self.manual_inspection_mode = True
            self.latest_frame = None
            self.status_var.set(f"Modo Inspeção Manual ativado - Câmera {camera_index} ativa")
            self.process_live_frame()
        except Exception as e:
            print(f"Erro ao iniciar inspeção manual: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar inspeção manual: {str(e)}")

    def stop_live_capture_manual_inspection(self):
        """Para a captura contínua com inspeção manual."""
        self.live_capture = False
        self.manual_inspection_mode = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        self.status_var.set("Modo Inspeção Manual desativado")
    
    def clear_all(self):
        """Limpa todos os dados do editor."""
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        self.slots = []
        self.selected_slot_id = None
        self.model_path = None
        self.drawing = False
        self.current_rect = None
        self.current_model = None
        
        # Reset das flags de controle
        self._selecting_slot = False
        self._processing_edit_click = False
        
        # Limpa canvas
        self.canvas.delete("all")
        
        # Limpa informações do slot
        self.slot_info_label.config(text="Nenhum slot selecionado")
        
        # Atualiza status
        self.status_var.set("Dados limpos")
        self.update_button_states()
    
    def load_image_data(self, image_path):
        """Carrega dados da imagem e calcula escala."""
        try:
            # Carrega imagem
            self.img_original = cv2.imread(str(image_path))
            if self.img_original is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            print(f"Imagem carregada: {image_path}")
            print(f"Dimensões: {self.img_original.shape}")
            
            # Converte para exibição no canvas
            self.img_display, self.scale_factor = cv2_to_tk(self.img_original, PREVIEW_W, PREVIEW_H)
            
            if self.img_display is None:
                raise ValueError("Erro ao converter imagem para exibição")
            
            print(f"Escala aplicada: {self.scale_factor:.3f}")
            
            # Configura canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.img_display)
            
            # Atualiza região de scroll
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            return True
            
        except Exception as e:
            print(f"Erro em load_image_data: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
            return False
    
    def load_image(self):
        """Carrega uma nova imagem."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            # Preserva o modelo atual se estiver em criação
            current_model_backup = None
            if hasattr(self, 'current_model') and self.current_model and self.current_model.get('id') is None:
                current_model_backup = self.current_model.copy()
            
            self.clear_all()
            
            # Restaura o modelo em criação
            if current_model_backup:
                self.current_model = current_model_backup
                self.current_model['image_path'] = file_path  # Atualiza o caminho da imagem
            
            if self.load_image_data(file_path):
                self.status_var.set(f"Imagem carregada: {Path(file_path).name}")
                self.update_button_states()
    
    def auto_start_webcam(self):
        """Inicia automaticamente a webcam em segundo plano se houver câmeras disponíveis."""
        try:
            if self.available_cameras and len(self.available_cameras) > 0:
                # Seleciona a primeira câmera disponível
                self.camera_combo.set(str(self.available_cameras[0]))
                # Inicia a câmera em segundo plano (sem exibir no canvas)
                self.start_background_camera()
                print(f"Webcam iniciada em segundo plano: Câmera {self.available_cameras[0]}")
            else:
                print("Nenhuma câmera disponível para inicialização automática")
        except Exception as e:
            print(f"Erro na inicialização automática da webcam: {e}")
    
    def start_background_camera(self):
        """Inicia a câmera em segundo plano para captura quando solicitado."""
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução padrão para inicialização rápida
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Marca que a câmera está disponível em segundo plano
            self.live_capture = False  # Não está fazendo live capture
            self.live_view = False     # Não está exibindo no canvas
            
            self.status_var.set(f"Câmera {camera_index} pronta em segundo plano")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera em segundo plano: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
    
    def on_canvas_zoom(self, event):
        """Implementa zoom no canvas com a roda do mouse."""
        try:
            # Determinar direção do zoom
            if event.delta > 0:
                # Zoom in
                zoom_factor = 1.1
            else:
                # Zoom out
                zoom_factor = 0.9
            
            # Aplicar zoom
            old_zoom = self.zoom_level
            self.zoom_level *= zoom_factor
            
            # Limitar zoom entre 0.1x e 5.0x
            self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
            
            # Se o zoom mudou, redimensionar a imagem
            if self.zoom_level != old_zoom and hasattr(self, 'current_image') and self.current_image is not None:
                self.update_canvas_image()
                
        except Exception as e:
            print(f"Erro no zoom: {e}")
    
    def on_canvas_pan_start(self, event):
        """Inicia o pan com o botão do meio do mouse."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
    
    def on_canvas_pan_drag(self, event):
        """Executa o pan arrastando com o botão do meio."""
        try:
            # Calcular deslocamento
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # Mover a visualização do canvas
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            
        except Exception as e:
            print(f"Erro no pan: {e}")
    
    def on_canvas_pan_end(self, event):
        """Finaliza o pan."""
        self.canvas.config(cursor="")
    
    def update_canvas_image(self):
        """Atualiza a imagem no canvas com o nível de zoom atual."""
        try:
            if hasattr(self, 'current_image') and self.current_image is not None:
                # Calcular novo tamanho
                original_height, original_width = self.current_image.shape[:2]
                new_width = int(original_width * self.zoom_level)
                new_height = int(original_height * self.zoom_level)
                
                # Redimensionar imagem
                resized_image = cv2.resize(self.current_image, (new_width, new_height))
                
                # Converter para formato do Tkinter
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                self.photo = ImageTk.PhotoImage(image_pil)
                
                # Atualizar canvas
                self.canvas.delete("image")
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="image")
                
                # Atualizar região de scroll
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
                # Redesenhar slots
                self.draw_slots()
                
        except Exception as e:
            print(f"Erro ao atualizar imagem do canvas: {e}")
    
    def start_live_capture(self):
        """Inicia captura contínua da câmera em segundo plano."""
        if self.live_capture:
            return
            
        try:
            # Desativa outros modos de captura se estiverem ativos
            if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
                self.stop_live_capture_manual_inspection()
                
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Para live view se estiver ativo
            if self.live_view:
                self.stop_live_view()
                
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance e inicialização rápida
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            # Garante que o modo de inspeção manual está desativado
            self.manual_inspection_mode = False
            self.process_live_frame()
            self.status_var.set(f"Câmera {camera_index} ativa em segundo plano")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
    
    def stop_live_capture(self):
        """Para a captura contínua da câmera."""
        self.live_capture = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        self.status_var.set("Câmera desconectada")
    
    def toggle_live_capture(self):
        """Alterna entre iniciar e parar a captura contínua."""
        if not self.live_capture:
            self.start_live_capture()
            if self.live_capture:  # Se iniciou com sucesso
                self.btn_live_capture.config(text="Parar Captura Contínua")
                # Reseta os outros botões de captura
                # btn_continuous_inspect e btn_manual_inspect removidos
        else:
            self.stop_live_capture()
            self.btn_live_capture.config(text="Iniciar Captura Contínua")
    
    def process_live_frame(self):
        """Processa frames da câmera em segundo plano."""
        if not self.live_capture or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            self.stop_live_capture()
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if self.live_capture:
            self.master.after(100, self.process_live_frame)
    
    def capture_from_webcam(self):
        """Captura instantânea da imagem mais recente da câmera."""
        try:
            if not self.live_capture or self.latest_frame is None:
                # Fallback para captura única se não há captura contínua
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                captured_image = capture_image_from_camera(camera_index)
            else:
                # Usa o frame mais recente da captura contínua
                captured_image = self.latest_frame.copy()
            
            if captured_image is not None:
                # Limpa dados anteriores
                self.clear_all()
                
                # Carrega a imagem capturada
                self.img_original = captured_image
                
                # Converte para exibição
                self.img_display, self.scale_factor = cv2_to_tk(self.img_original, PREVIEW_W, PREVIEW_H)
                
                if self.img_display:
                    # Limpa o canvas e exibe a nova imagem
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=NW, image=self.img_display)
                    
                    # Atualiza a região de scroll
                    self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                    
                    # Atualiza estado dos botões
                    self.update_button_states()
                    
                    camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                    self.status_var.set(f"Imagem capturada da câmera {camera_index}")
                    messagebox.showinfo("Sucesso", "Imagem capturada instantaneamente!")
                else:
                    messagebox.showerror("Erro", "Erro ao processar a imagem capturada.")
            else:
                messagebox.showerror("Erro", "Nenhuma imagem disponível para captura.")
                
        except Exception as e:
            print(f"Erro ao capturar da webcam: {e}")
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
    
    def load_model_dialog(self):
        """Abre diálogo para carregar modelo do banco de dados."""
        dialog = ModelSelectorDialog(self.master, self.db_manager)
        result = dialog.show()
        
        if result:
            if result['action'] == 'load':
                self.load_model_from_db(result['model_id'])
            elif result['action'] == 'new':
                self.create_new_model(result['name'])
    
    def load_model_from_db(self, model_id):
        """Carrega um modelo do banco de dados."""
        try:
            # Carrega dados do modelo
            model_data = self.db_manager.load_modelo(model_id)
            
            # Limpa dados atuais
            self.clear_all()
            
            # Carrega imagem de referência
            image_path = model_data['image_path']
            
            # Tenta caminho absoluto primeiro
            if not Path(image_path).exists():
                # Tenta caminho relativo ao diretório de modelos
                relative_path = MODEL_DIR / Path(image_path).name
                if relative_path.exists():
                    image_path = str(relative_path)
                else:
                    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
            
            if not self.load_image_data(image_path):
                return
            
            # Carrega slots
            self.slots = model_data['slots']
            self.current_model_id = model_id
            # Define o modelo atual para uso em outras funções
            self.current_model = model_data
            
            # Atualiza interface
            self.update_slots_list()
            self.redraw_slots()
            
            self.status_var.set(f"Modelo carregado: {model_data['nome']} ({len(self.slots)} slots)")
            self.update_button_states()
            
            # Marca modelo como salvo (recém carregado)
            self.mark_model_saved()
            
            print(f"Modelo '{model_data['nome']}' carregado com sucesso: {len(self.slots)} slots")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.status_var.set(f"Erro ao carregar modelo: {str(e)}")
    
    def create_new_model(self, model_name):
        """Cria um novo modelo vazio."""
        try:
            # Limpa dados atuais
            self.clear_all()
            
            # Define como novo modelo (sem ID ainda)
            self.current_model_id = None
            self.slots = []
            
            # Define dados temporários do modelo para permitir adição de slots
            # IMPORTANTE: Deve ser definido APÓS clear_all() que limpa current_model
            self.current_model = {
                'id': None,  # Será definido quando salvar
                'nome': model_name,
                'image_path': self.image_path if hasattr(self, 'image_path') else None,
                'slots': []
            }
            
            self.status_var.set(f"Novo modelo criado: {model_name}")
            self.update_button_states()
            
            # Marca modelo como salvo (novo modelo vazio)
            self.mark_model_saved()
            
            print(f"Novo modelo '{model_name}' criado")
            
        except Exception as e:
            print(f"Erro ao criar novo modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao criar novo modelo: {str(e)}")
    
    def update_slots_tree(self):
        """Atualiza as informações de slots na interface."""
        try:
            # Atualiza a informação do slot selecionado
            self.update_slot_info_display()
        except Exception as e:
            print(f"Erro ao atualizar informações de slots: {e}")
            import traceback
            traceback.print_exc()
            
    def update_slot_info_display(self):
        """Atualiza o display de informações do slot selecionado."""
        if self.selected_slot_id is None:
            self.slot_info_label.config(text="Nenhum slot selecionado")
            return
            
        # Busca o slot selecionado
        selected_slot = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
        if not selected_slot:
            self.slot_info_label.config(text=f"Erro: Slot {self.selected_slot_id} não encontrado")
            return
            
        # Formata as informações do slot
        info_text = f"ID: {selected_slot['id']}\n"
        info_text += f"Tipo: {selected_slot.get('tipo', 'N/A')}\n"
        info_text += f"Posição: ({selected_slot.get('x', 0)}, {selected_slot.get('y', 0)})\n"
        info_text += f"Tamanho: {selected_slot.get('w', 0)}x{selected_slot.get('h', 0)}"
        
        # Adiciona informações específicas do tipo de slot
        if selected_slot.get('tipo') == 'clip':
            info_text += f"\nLimiar: {selected_slot.get('detection_threshold', 0.8)}"
            
        self.slot_info_label.config(text=info_text)
    
    def update_slots_list(self):
        """Função legada para compatibilidade - redireciona para update_slots_tree."""
        self.update_slots_tree()
    
    def redraw_slots(self):
        """Redesenha todos os slots no canvas."""
        try:
            if self.img_display is None or not hasattr(self, 'canvas'):
                return
            
            # Remove retângulos existentes
            self.canvas.delete("slot")
            
            # Desenha cada slot
            for slot in self.slots:
                if slot and 'id' in slot:
                    self.draw_slot(slot)
        except Exception as e:
            print(f"Erro ao redesenhar slots: {e}")
            self.status_var.set("Erro ao atualizar visualização")
    
    def draw_slot(self, slot):
        """Desenha um slot no canvas."""
        try:
            # Verifica se o slot tem todos os campos necessários
            required_fields = ['x', 'y', 'w', 'h', 'id', 'tipo']
            if not all(field in slot for field in required_fields):
                print(f"Slot inválido: campos obrigatórios ausentes {slot}")
                return
            
            # Verifica se scale_factor existe
            if not hasattr(self, 'scale_factor') or self.scale_factor <= 0:
                print("Scale factor inválido")
                return
            
            # Converte coordenadas da imagem para canvas (incluindo offsets)
            x1 = int(slot['x'] * self.scale_factor) + self.x_offset
            y1 = int(slot['y'] * self.scale_factor) + self.y_offset
            x2 = int((slot['x'] + slot['w']) * self.scale_factor) + self.x_offset
            y2 = int((slot['y'] + slot['h']) * self.scale_factor) + self.y_offset
            
            # Calcula centro para rotação
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Escolhe cor baseada na seleção
            if slot['id'] == self.selected_slot_id:
                color = style_config["selection_color"]
                width = 3
            else:
                color = COLOR_CLIP
                width = 2
            
            # Obtém rotação do slot
            # Desenha retângulo simples (rotação removida)
            shape_id = self.canvas.create_rectangle(x1, y1, x2, y2, 
                                       outline=color, width=width, tags="slot")
            
            # Desenha áreas de exclusão se existirem
            exclusion_areas = slot.get('exclusion_areas', [])
            for exclusion in exclusion_areas:
                ex_x1 = int(exclusion['x'] * self.scale_factor) + self.x_offset
                ex_y1 = int(exclusion['y'] * self.scale_factor) + self.y_offset
                ex_x2 = int((exclusion['x'] + exclusion['w']) * self.scale_factor) + self.x_offset
                ex_y2 = int((exclusion['y'] + exclusion['h']) * self.scale_factor) + self.y_offset
                
                # Desenha área de exclusão em vermelho
                self.canvas.create_rectangle(ex_x1, ex_y1, ex_x2, ex_y2,
                                            outline="#FF4444", width=2, tags="slot")
            
            # Adiciona texto com ID (já usando x1, y1 corrigidos com offsets)
            # Carrega as configurações de estilo
            style_config = load_style_config()
            self.canvas.create_text(x1 + 5, y1 + 5, text=slot['id'],
                                   fill="white", font=style_config["ok_font"], tags="slot")
            
            # Adiciona botão de edição (pequeno quadrado no canto superior direito)
            edit_size = 12
            edit_x1 = x2 - edit_size - 2
            edit_y1 = y1 + 2
            edit_x2 = x2 - 2
            edit_y2 = y1 + edit_size + 2
            
            edit_btn = self.canvas.create_rectangle(edit_x1, edit_y1, edit_x2, edit_y2,
                                                   fill="#4CAF50", outline="white", width=1,
                                                   tags=("slot", f"edit_btn_{slot['id']}"))
            
            # Adiciona ícone de edição (pequeno "E")
            # Carrega as configurações de estilo se ainda não foi carregado
            if 'style_config' not in locals():
                style_config = load_style_config()
            self.canvas.create_text((edit_x1 + edit_x2) // 2, (edit_y1 + edit_y2) // 2,
                                   text="E", fill="white", font=style_config["ok_font"],
                                   tags=("slot", f"edit_text_{slot['id']}"))
        except Exception as e:
            print(f"Erro ao desenhar slot {slot.get('id', 'desconhecido')}: {e}")
    
    def on_canvas_press(self, event):
        """Inicia desenho de novo slot ou edita slot existente."""
        try:
            if self.img_original is None:
                return
            
            # Verifica se o canvas existe e está válido
            if not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
                print("Canvas não existe ou foi destruído")
                return
            
            # Converte coordenadas do canvas para coordenadas da tela
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Verifica se clicou em um handle de edição primeiro
            try:
                closest_items = self.canvas.find_closest(canvas_x, canvas_y)
                if closest_items:
                    clicked_item = closest_items[0]
                    tags = self.canvas.gettags(clicked_item)
                    
                    # Verifica se é um handle de edição
                    for tag in tags:
                        if tag == "edit_handle" or tag.startswith("resize_handle_"):
                            # Deixa o evento ser processado pelos handles
                            return
            except Exception as e:
                print(f"Erro ao verificar handles: {e}")
            
            # Verifica se clicou em um botão de edição
            try:
                closest_items = self.canvas.find_closest(canvas_x, canvas_y)
                if not closest_items:
                    print("Nenhum item encontrado no canvas")
                    return
                
                clicked_item = closest_items[0]
                tags = self.canvas.gettags(clicked_item)
                
                if not tags:
                    print("Item clicado não possui tags")
                    # Continua para verificar slots existentes
                else:
                    for tag in tags:
                        if tag.startswith('edit_btn_') or tag.startswith('edit_text_'):
                            try:
                                # Extrai o slot_id da tag
                                tag_parts = tag.split('_')
                                if len(tag_parts) < 3:
                                    print(f"Tag inválida: {tag}")
                                    continue
                                
                                slot_id = int(tag_parts[-1])
                                
                                # Verifica se o slot existe
                                if not any(s['id'] == slot_id for s in self.slots):
                                    print(f"Slot {slot_id} não encontrado na lista")
                                    return
                                
                                # Previne múltiplas chamadas simultâneas
                                if hasattr(self, '_processing_edit_click') and self._processing_edit_click:
                                    print("Já processando clique de edição")
                                    return
                                
                                self._processing_edit_click = True
                                
                                try:
                                    print(f"Processando clique no botão de edição do slot {slot_id}")
                                    self.select_slot(slot_id)
                                    # Removido chamada automática para edit_selected_slot() para evitar travamento
                                    print(f"Slot {slot_id} selecionado. Use o botão 'Editar Slot Selecionado' para editar.")
                                    return
                                finally:
                                    self._processing_edit_click = False
                                    
                            except ValueError as ve:
                                print(f"Erro ao converter slot_id: {ve}")
                                continue
                            except Exception as e:
                                print(f"Erro ao processar clique no botão de edição: {e}")
                                import traceback
                                traceback.print_exc()
                                return
                            
            except Exception as e:
                 print(f"Erro ao verificar item clicado: {e}")
                 import traceback
                 traceback.print_exc()
            
            # Verifica se clicou em um slot existente
            try:
                clicked_slot = self.find_slot_at(canvas_x, canvas_y)
                if clicked_slot:
                    print(f"Clicou no slot {clicked_slot['id']}")
                    
                    # Se está no modo de exclusão e há um slot selecionado, permite desenhar área de exclusão
                    if self.current_drawing_mode == "exclusion" and self.selected_slot_id is not None:
                        print("Iniciando desenho de área de exclusão")
                        self.drawing = True
                        self.start_x = canvas_x
                        self.start_y = canvas_y
                        self.canvas.delete("drawing")
                        return
                    else:
                        # Seleciona o slot e mostra handles de edição
                        self.select_slot(clicked_slot['id'])
                        self.show_edit_handles(clicked_slot)
                        return
            except Exception as e:
                print(f"Erro ao verificar slot existente: {e}")
                import traceback
                traceback.print_exc()
            
            # Se está no modo de exclusão mas não há slot selecionado, mostra mensagem
            if self.current_drawing_mode == "exclusion":
                self.status_var.set("Selecione um slot primeiro para criar área de exclusão")
                return
            
            # Inicia desenho de novo slot
            try:
                print("Iniciando desenho de novo slot")
                self.deselect_all_slots()
                self.hide_edit_handles()
                self.drawing = True
                self.start_x = canvas_x
                self.start_y = canvas_y
                
                # Remove retângulo de desenho anterior
                self.canvas.delete("drawing")
            except Exception as e:
                print(f"Erro ao iniciar desenho de novo slot: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Erro geral em on_canvas_press: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erro ao processar clique no canvas")
    
    def on_canvas_drag(self, event):
        """Atualiza desenho do slot durante arraste."""
        if not self.drawing:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Remove forma anterior
        self.canvas.delete("drawing")
        
        # Define cor baseada no modo
        if self.current_drawing_mode == "exclusion":
            outline_color = "#FF4444"  # Vermelho para exclusão
        else:
            outline_color = COLOR_DRAWING
        
        # Desenha retângulo (para rectangle e exclusion)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, canvas_x, canvas_y,
            outline=outline_color, width=2, tags="drawing"
        )
    
    def on_canvas_release(self, event):
        """Finaliza desenho do slot."""
        if not self.drawing:
            return
        
        self.drawing = False
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Remove forma de desenho
        self.canvas.delete("drawing")
        
        # Calcula dimensões
        x1, y1 = min(self.start_x, canvas_x), min(self.start_y, canvas_y)
        x2, y2 = max(self.start_x, canvas_x), max(self.start_y, canvas_y)
        
        w = x2 - x1
        h = y2 - y1
        
        # Verifica se a área é válida
        if w < 10 or h < 10:
            self.status_var.set("Área muito pequena (mínimo 10x10 pixels)")
            return
        
        # Converte coordenadas do canvas para imagem original
        img_x = int(x1 / self.scale_factor)
        img_y = int(y1 / self.scale_factor)
        img_w = int(w / self.scale_factor)
        img_h = int(h / self.scale_factor)
        
        # Verifica se é área de exclusão
        if self.current_drawing_mode == "exclusion":
            self.add_exclusion_area(img_x, img_y, img_w, img_h)
        else:
            # Adiciona slot normal
            self.add_slot(img_x, img_y, img_w, img_h)
    
    def find_slot_at(self, canvas_x, canvas_y):
        """Encontra slot nas coordenadas do canvas."""
        for slot in self.slots:
            x1 = slot['x'] * self.scale_factor
            y1 = slot['y'] * self.scale_factor
            x2 = (slot['x'] + slot['w']) * self.scale_factor
            y2 = (slot['y'] + slot['h']) * self.scale_factor
            
            # Verificação simples de retângulo
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return slot
                    
        return None
    
    def select_slot(self, slot_id):
        """Seleciona um slot."""
        try:
            # Previne loop infinito
            if hasattr(self, '_selecting_slot') and self._selecting_slot:
                return
            
            self._selecting_slot = True
            
            # Verifica se o slot existe
            slot_exists = any(s['id'] == slot_id for s in self.slots)
            if not slot_exists:
                print(f"Erro: Slot {slot_id} não encontrado")
                return
            
            self.selected_slot_id = slot_id
            
            # Atualiza informações do slot selecionado
            self.update_slot_info_display()
            
            # Mostra automaticamente o editor de slot no painel direito
            slot_to_edit = next((s for s in self.slots if s['id'] == slot_id), None)
            if slot_to_edit:
                self.show_slot_editor_in_right_panel(slot_to_edit)
            
            self.redraw_slots()
            self.update_button_states()
            self.status_var.set(f"Slot {slot_id} selecionado - Editor aberto no painel direito")
        except Exception as e:
            print(f"Erro ao selecionar slot {slot_id}: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erro na seleção do slot")
        finally:
            self._selecting_slot = False
    
    def deselect_all_slots(self):
        """Remove seleção de todos os slots."""
        self.selected_slot_id = None
        self.slot_info_label.config(text="Nenhum slot selecionado")
        
        # Exibe mensagem padrão no painel direito quando nenhum slot está selecionado
        self.show_default_right_panel()
        
        self.hide_edit_handles()
        self.redraw_slots()
        self.update_button_states()
    
    def add_slot(self, xa, ya, wa, ha):
        """Adiciona um novo slot."""
        if self.img_original is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return
        
        # Converte coordenadas do canvas para imagem original
        x = int(xa)
        y = int(ya)
        w = int(wa)
        h = int(ha)
        
        # Valida coordenadas
        img_h, img_w = self.img_original.shape[:2]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            messagebox.showerror("Erro", "Slot está fora dos limites da imagem.")
            return
        
        # Extrai ROI
        roi = self.img_original[y:y+h, x:x+w]
        if roi.size == 0:
            messagebox.showerror("Erro", "ROI do slot está vazia.")
            return
        
        print(f"add_slot: Adicionando slot na posição ({x}, {y}), tamanho ({w}, {h})")
        
        # Apenas slots do tipo 'clip' são suportados
        slot_type = 'clip'
        
        # Valores padrão (não utilizados para clips, mas mantidos para compatibilidade)
        bgr_color = [0, 0, 255]  # Vermelho padrão
        h_tolerance = 10
        s_tolerance = 50
        v_tolerance = 50
        
        # Gera ID único
        existing_ids = [slot['id'] for slot in self.slots]
        slot_id = 1
        while slot_id in existing_ids:
            slot_id += 1
        
        # Cria dados do slot com configurações padrão específicas por tipo
        slot_data = {
            'id': slot_id,
            'tipo': slot_type,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cor': bgr_color,
            'h_tolerance': h_tolerance,
            's_tolerance': s_tolerance,
            'v_tolerance': v_tolerance,
            'detection_threshold': 0.8,  # Limiar padrão para detecção
            'shape': self.current_drawing_mode,  # Forma: rectangle, exclusion
            'exclusion_areas': []               # Lista de áreas de exclusão
        }
        
        # Configurações específicas para clips
        slot_data.update({
            'correlation_threshold': THR_CORR,
            'template_method': 'TM_CCOEFF_NORMED',
            'scale_tolerance': 0.1
        })
        
        # Armazena o ROI em memória para salvar depois quando o modelo for salvo
        slot_data['roi_data'] = roi.copy()  # Armazena uma cópia do ROI
        slot_data['template_filename'] = f"slot_{slot_id}_template.png"
        
        print(f"ROI do slot {slot_id} armazenado em memória para salvamento posterior")
        
        # Adiciona slot à lista
        self.slots.append(slot_data)
        
        # Atualiza interface
        self.update_slots_list()
        self.redraw_slots()
        
        self.status_var.set(f"Slot {slot_id} ({slot_type}) adicionado")
        self.update_button_states()
        
        # Marca modelo como modificado
        self.mark_model_modified()
        
        print(f"Slot {slot_id} adicionado com sucesso: {slot_data}")
    
    # Funções on_slot_select e on_slot_double_click foram removidas
    # pois não são mais necessárias sem o slots_listbox
    
    def clear_slots(self):
        """Remove todos os slots."""
        if not self.slots:
            messagebox.showinfo("Info", "Nenhum slot para remover.")
            return
        
        if messagebox.askyesno("Confirmar", f"Remover todos os {len(self.slots)} slots?"):
            self.slots = []
            self.selected_slot_id = None
            self.update_slots_list()
            self.redraw_slots()
            self.status_var.set("Todos os slots removidos")
            self.update_button_states()
            
            # Marca modelo como modificado
            self.mark_model_modified()
    
    # Função edit_selected_slot removida - editor aparece automaticamente quando slot é selecionado
    
    def clear_right_panel(self):
        """Limpa o painel direito."""
        for widget in self.right_panel.winfo_children():
            widget.destroy()
    
    def show_default_right_panel(self):
        """Exibe mensagem padrão no painel direito quando nenhum slot está selecionado."""
        self.clear_right_panel()
        
        # Título do painel
        title_label = ttk.Label(self.right_panel, text="Editor de Slot", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Mensagem informativa
        info_frame = ttk.Frame(self.right_panel)
        info_frame.pack(fill=X, pady=10)
        
        info_label = ttk.Label(info_frame, 
                              text="Selecione um slot no\nEditor de Malha para\neditar suas propriedades",
                              justify=CENTER,
                              foreground="gray")
        info_label.pack()
        
        # Instruções
        instructions_frame = ttk.LabelFrame(self.right_panel, text="Instruções")
        instructions_frame.pack(fill=X, pady=(20, 0))
        
        instructions_text = (
            "• Clique em um slot no canvas\n"
            "  para selecioná-lo\n\n"
            "• O editor aparecerá\n"
            "  automaticamente\n\n"
            "• Ajuste posição, tamanho\n"
            "  e parâmetros de detecção"
        )
        
        instructions_label = ttk.Label(instructions_frame, 
                                     text=instructions_text,
                                     justify=LEFT,
                                     foreground="#666666")
        instructions_label.pack(padx=10, pady=10)
    
    def save_slot_changes(self, slot_data):
        """Salva as alterações feitas no slot"""
        try:
            # Obtém os valores dos campos
            new_x = int(self.edit_vars['x'].get())
            new_y = int(self.edit_vars['y'].get())
            new_w = int(self.edit_vars['w'].get())
            new_h = int(self.edit_vars['h'].get())
            
            # Atualiza os dados do slot
            for slot in self.slots:
                if slot['id'] == slot_data['id']:
                    slot['x'] = new_x
                    slot['y'] = new_y
                    slot['w'] = new_w
                    slot['h'] = new_h
                    
                    # Para slots do tipo clip, atualiza parâmetros de detecção
                    if slot.get('tipo') == 'clip' and 'detection_method' in self.edit_vars:
                        # Método de detecção
                        old_method = slot.get('detection_method', 'template_matching')
                        new_method = self.edit_vars['detection_method'].get()
                        
                        # Atualiza o método de detecção
                        slot['detection_method'] = new_method
                        print(f"Método de detecção alterado de {old_method} para {new_method}")
                        
                        # Limiar de detecção
                        slot['detection_threshold'] = float(self.edit_vars['detection_threshold'].get())
                        
                        # Porcentagem para OK
                        if 'ok_threshold' in self.edit_vars:
                            slot['ok_threshold'] = int(self.edit_vars['ok_threshold'].get())
                        
                        # Limiar de correlação
                        if 'correlation_threshold' in self.edit_vars:
                            slot['correlation_threshold'] = float(self.edit_vars['correlation_threshold'].get())
                    
                    # Salva no banco de dados se há um modelo carregado
                    if self.current_model_id is not None:
                        try:
                            self.db_manager.update_slot(slot['db_id'], slot)
                        except Exception as e:
                            print(f"Erro ao salvar slot no banco: {e}")
                            messagebox.showwarning("Aviso", "Slot atualizado na interface, mas não foi salvo no banco de dados.")
                    
                    break
            
            # Redesenha o canvas
            self.redraw_slots()
            
            # Atualiza a lista de slots
            self.update_slots_list()
            
            # Limpa o painel direito
            self.clear_right_panel()
            
            # Marca o modelo como modificado
            self.mark_model_modified()
            
            # Atualiza a mensagem de status
            self.status_var.set(f"Slot {slot_data['id']} atualizado com sucesso")
            
            print(f"Slot {slot_data['id']} atualizado com sucesso")
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Valores inválidos: {str(e)}\nVerifique se todos os campos contêm números válidos.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
    def edit_slot_with_simple_dialogs(self, slot_data):
        """Edita o slot usando diálogos simples do tkinter"""
        from tkinter import simpledialog
        
        print(f"Editando slot {slot_data['id']} com diálogos simples")
        
        # Edita X
        new_x = simpledialog.askinteger(
            "Editar Slot", 
            f"Posição X atual: {slot_data['x']}\nNova posição X:",
            initialvalue=slot_data['x'],
            minvalue=0
        )
        if new_x is None:  # Usuário cancelou
            return
        
        # Edita Y
        new_y = simpledialog.askinteger(
            "Editar Slot", 
            f"Posição Y atual: {slot_data['y']}\nNova posição Y:",
            initialvalue=slot_data['y'],
            minvalue=0
        )
        if new_y is None:
            return
        
        # Edita Largura
        new_w = simpledialog.askinteger(
            "Editar Slot", 
            f"Largura atual: {slot_data['w']}\nNova largura:",
            initialvalue=slot_data['w'],
            minvalue=1
        )
        if new_w is None:
            return
        
        # Edita Altura
        new_h = simpledialog.askinteger(
            "Editar Slot", 
            f"Altura atual: {slot_data['h']}\nNova altura:",
            initialvalue=slot_data['h'],
            minvalue=1
        )
        if new_h is None:
            return
        
        # Para slots do tipo clip, edita o limiar de detecção
        new_threshold = None
        if slot_data.get('tipo') == 'clip':
            current_threshold = slot_data.get('detection_threshold', 0.8)
            new_threshold = simpledialog.askfloat(
                "Editar Slot", 
                f"Limiar de detecção atual: {current_threshold}\nNovo limiar (0.0 - 1.0):",
                initialvalue=current_threshold,
                minvalue=0.0,
                maxvalue=1.0
            )
            if new_threshold is None:
                return
        
        # Aplica as alterações
        slot_data['x'] = new_x
        slot_data['y'] = new_y
        slot_data['w'] = new_w
        slot_data['h'] = new_h
        
        if new_threshold is not None:
            slot_data['detection_threshold'] = new_threshold
        
        # Atualiza a exibição
        self.redraw_slots()
        self.update_slots_list()
        
        print(f"Slot {slot_data['id']} atualizado: X={new_x}, Y={new_y}, W={new_w}, H={new_h}")
        if new_threshold is not None:
            print(f"Limiar de detecção: {new_threshold}")
        
        messagebox.showinfo("Sucesso", f"Slot {slot_data['id']} atualizado com sucesso!")
    
    def show_slot_editor_in_right_panel(self, slot_data):
        """Cria um editor de slot simplificado no painel direito"""
        print("Criando editor de slot no painel direito...")
        
        # Carrega as configurações de estilo
        self.style_config = load_style_config()
        
        # Limpa o painel direito
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Título do editor
        title_frame = ttk.Frame(self.right_panel)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text=f"Editar Slot {slot_data['id']}", 
                               font=('Arial', 10, 'bold'),
                               foreground=self.style_config["text_color"])
        title_label.pack(pady=(0, 5))
        
        # Frame com scrollbar para os campos
        editor_frame = ttk.Frame(self.right_panel)
        editor_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Variáveis para os campos
        self.edit_vars = {}
        
        # Seção: Posição e Tamanho
        position_frame = ttk.LabelFrame(editor_frame, text="Posição e Tamanho")
        position_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        # Campos básicos com descrições simples
        basic_fields = [
            ('X:', 'x', slot_data['x'], "Posição horizontal"),
            ('Y:', 'y', slot_data['y'], "Posição vertical"),
            ('Largura:', 'w', slot_data['w'], "Largura do slot"),
            ('Altura:', 'h', slot_data['h'], "Altura do slot")
        ]
        
        for i, (label_text, key, value, tooltip) in enumerate(basic_fields):
            row_frame = ttk.Frame(position_frame)
            row_frame.pack(fill='x', pady=2, padx=5)
            
            label = ttk.Label(row_frame, text=label_text, width=8)
            label.pack(side='left')
            
            var = StringVar(value=str(value))
            self.edit_vars[key] = var
            entry = ttk.Entry(row_frame, textvariable=var, width=8)
            entry.pack(side='left', padx=(5, 0))
            
            # Tooltip simples
            tip_label = ttk.Label(row_frame, text=tooltip, font=("Arial", 8), foreground="#888888")
            tip_label.pack(side='left', padx=(5, 0))
        
        # Seção: Detecção (para slots do tipo clip)
        if slot_data.get('tipo') == 'clip':
            detection_frame = ttk.LabelFrame(editor_frame, text="Parâmetros de Detecção")
            detection_frame.pack(fill='x', pady=(0, 10), padx=5)
            
            # Método de detecção
            method_frame = ttk.Frame(detection_frame)
            method_frame.pack(fill='x', pady=2, padx=5)
            
            method_label = ttk.Label(method_frame, text="Método:", width=8)
            method_label.pack(side='left')
            
            detection_methods = [
                "template_matching", # Comparação de imagem
                "histogram_analysis", # Análise de histograma
                "contour_analysis", # Análise de contorno
                "image_comparison" # Comparação direta de imagem
            ]
            method_var = StringVar(value=slot_data.get('detection_method', 'template_matching'))
            self.edit_vars['detection_method'] = method_var
            
            method_combo = ttk.Combobox(method_frame, textvariable=method_var, 
                                      values=detection_methods, width=15, state="readonly")
            method_combo.pack(side='left', padx=(5, 0))
            
            # Tooltip para explicar cada método
            method_tip = ttk.Label(method_frame, text="Selecione o método de detecção", 
                                  font=("Arial", 8), foreground="#888888")
            method_tip.pack(side='left', padx=(5, 0))
            
            # Atualiza o tooltip baseado na seleção
            def update_method_tip(*args):
                method = method_var.get()
                if method == "template_matching":
                    method_tip.config(text="Comparação de imagem com template")
                elif method == "histogram_analysis":
                    method_tip.config(text="Análise de distribuição de cores")
                elif method == "contour_analysis":
                    method_tip.config(text="Análise de contornos e formas")
                elif method == "image_comparison":
                    method_tip.config(text="Comparação direta entre imagens")
            
            method_var.trace("w", update_method_tip)
            update_method_tip()  # Inicializa o tooltip
            
            # Adiciona um canvas para preview do filtro
            preview_frame = ttk.LabelFrame(detection_frame, text="Preview do Filtro")
            preview_frame.pack(fill='x', pady=5, padx=5)
            
            # Canvas para exibir o preview
            self.preview_canvas = Canvas(preview_frame, bg="#1E1E1E", width=200, height=150)
            self.preview_canvas.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Definir variáveis antes da função update_preview_filter
            threshold_var = StringVar(value=str(slot_data.get('detection_threshold', 0.8)))
            ok_threshold_var = StringVar(value=str(slot_data.get('ok_threshold', 70)))
            
            # Função para atualizar o preview quando o método de detecção mudar
            def update_preview_filter(*args):
                if not hasattr(self, 'img_original') or self.img_original is None:
                    return
                
                # Obtém o slot atual
                if not slot_data or 'x' not in slot_data or 'y' not in slot_data or 'w' not in slot_data or 'h' not in slot_data:
                    return
                
                # Extrai a ROI do slot
                x, y, w, h = slot_data['x'], slot_data['y'], slot_data['w'], slot_data['h']
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > self.img_original.shape[1] or y + h > self.img_original.shape[0]:
                    return
                
                roi = self.img_original[y:y+h, x:x+w].copy()
                
                # Aplica o filtro selecionado
                method = method_var.get()
                filtered_roi = roi.copy()
                
                try:
                    if method == "histogram_analysis":
                        # Converte para HSV e visualiza o histograma
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        h_bins = 50
                        s_bins = 60
                        hist_range = [0, 180, 0, 256]  # H: 0-179, S: 0-255
                        hist = cv2.calcHist([roi_hsv], [0, 1], None, [h_bins, s_bins], hist_range)
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                        
                        # Cria uma visualização do histograma
                        hist_img = np.zeros((h_bins, s_bins), np.uint8)
                        for h in range(h_bins):
                            for s in range(s_bins):
                                hist_img[h, s] = min(255, int(hist[h, s] * 255))
                        
                        # Redimensiona para melhor visualização
                        hist_img = cv2.resize(hist_img, (w, h))
                        hist_img = cv2.applyColorMap(hist_img, cv2.COLORMAP_JET)
                        filtered_roi = hist_img
                        
                    elif method == "contour_analysis":
                        # Converte para escala de cinza e aplica detecção de bordas
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                        edges = cv2.Canny(roi_blur, 50, 150)
                        
                        # Encontra contornos
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Desenha contornos em uma imagem colorida
                        contour_img = np.zeros_like(roi)
                        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                        filtered_roi = contour_img
                        
                    elif method == "image_comparison":
                        # Verifica se há um template para comparação
                        template_path = slot_data.get('template_path')
                        if template_path and Path(template_path).exists():
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                # Redimensiona o template para o tamanho da ROI
                                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                                
                                # Calcula a diferença absoluta
                                diff = cv2.absdiff(roi, template_resized)
                                filtered_roi = diff
                        else:
                            # Se não há template, mostra mensagem
                            filtered_roi = np.zeros_like(roi)
                            cv2.putText(filtered_roi, "Sem template", (10, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    elif method == "template_matching":
                        # Verifica se há um template para comparação
                        template_path = slot_data.get('template_path')
                        if template_path and Path(template_path).exists():
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                # Mostra o template
                                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                                filtered_roi = template_resized
                                
                                # Adiciona texto indicando que é o template
                                cv2.putText(filtered_roi, "Template", (10, 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Se não há template, mostra mensagem
                            filtered_roi = np.zeros_like(roi)
                            cv2.putText(filtered_roi, "Sem template", (10, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Erro ao aplicar filtro: {e}")
                    filtered_roi = roi.copy()
                    cv2.putText(filtered_roi, "Erro no filtro", (10, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Adiciona informações sobre os parâmetros atuais
                try:
                    detection_threshold = float(threshold_var.get())
                    ok_threshold = int(ok_threshold_var.get())
                    
                    # Adiciona texto com os valores atuais
                    cv2.putText(filtered_roi, f"Limiar: {detection_threshold:.2f}", (10, h-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(filtered_roi, f"% OK: {ok_threshold}%", (10, h-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Erro ao adicionar informações ao preview: {e}")
                
                # Converte para exibição no canvas
                filtered_roi_rgb = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2RGB)
                filtered_roi_pil = Image.fromarray(filtered_roi_rgb)
                filtered_roi_tk = ImageTk.PhotoImage(filtered_roi_pil)
                
                # Atualiza o canvas
                self.preview_canvas.delete("all")
                
                # Usa as dimensões reais do canvas ou as dimensões configuradas se ainda não foi renderizado
                canvas_width = self.preview_canvas.winfo_width() if self.preview_canvas.winfo_width() > 1 else 200
                canvas_height = self.preview_canvas.winfo_height() if self.preview_canvas.winfo_height() > 1 else 150
                
                self.preview_canvas.create_image(canvas_width//2, 
                                               canvas_height//2, 
                                               image=filtered_roi_tk, anchor="center")
                self.preview_canvas.image = filtered_roi_tk  # Mantém referência
            
            # Vincula a função de atualização ao combobox
            method_var.trace("w", update_preview_filter)
            
            # Inicializa o preview
            update_preview_filter()
            
            # Limiar de detecção
            threshold_frame = ttk.Frame(detection_frame)
            threshold_frame.pack(fill='x', pady=2, padx=5)
            
            threshold_label = ttk.Label(threshold_frame, text="Limiar:", width=8)
            threshold_label.pack(side='left')
            
            self.edit_vars['detection_threshold'] = threshold_var
            threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=8)
            threshold_entry.pack(side='left', padx=(5, 0))
            
            # Vincula a função de atualização do preview ao limiar de detecção
            threshold_var.trace("w", update_preview_filter)
            
            threshold_tip = ttk.Label(threshold_frame, text="Valor entre 0.0 e 1.0", 
                                    font=("Arial", 8), foreground="#888888")
            threshold_tip.pack(side='left', padx=(5, 0))
            
            # Porcentagem para OK
            ok_threshold_frame = ttk.Frame(detection_frame)
            ok_threshold_frame.pack(fill='x', pady=2, padx=5)
            
            ok_threshold_label = ttk.Label(ok_threshold_frame, text="% para OK:", width=8)
            ok_threshold_label.pack(side='left')
            
            self.edit_vars['ok_threshold'] = ok_threshold_var
            ok_threshold_entry = ttk.Entry(ok_threshold_frame, textvariable=ok_threshold_var, width=8)
            ok_threshold_entry.pack(side='left', padx=(5, 0))
            
            # Vincula a função de atualização do preview à porcentagem para OK
            ok_threshold_var.trace("w", update_preview_filter)
            
            ok_threshold_tip = ttk.Label(ok_threshold_frame, text="Porcentagem para considerar OK (0-100)", 
                                       font=("Arial", 8), foreground="#888888")
            ok_threshold_tip.pack(side='left', padx=(5, 0))
            
            # Limiar de correlação
            correlation_threshold_frame = ttk.Frame(detection_frame)
            correlation_threshold_frame.pack(fill='x', pady=2, padx=5)
            
            correlation_threshold_label = ttk.Label(correlation_threshold_frame, text="Correlação:", width=8)
            correlation_threshold_label.pack(side='left')
            
            correlation_threshold_var = StringVar(value=str(slot_data.get('correlation_threshold', 0.5)))
            self.edit_vars['correlation_threshold'] = correlation_threshold_var
            correlation_threshold_entry = ttk.Entry(correlation_threshold_frame, textvariable=correlation_threshold_var, width=8)
            correlation_threshold_entry.pack(side='left', padx=(5, 0))
            
            # Vincula a função de atualização do preview ao limiar de correlação
            correlation_threshold_var.trace("w", update_preview_filter)
            
            correlation_threshold_tip = ttk.Label(correlation_threshold_frame, text="Limiar de correlação (0.0-1.0)", 
                                                 font=("Arial", 8), foreground="#888888")
            correlation_threshold_tip.pack(side='left', padx=(5, 0))
        
        # Botões de ação
        buttons_frame = ttk.Frame(self.right_panel)
        buttons_frame.pack(fill='x', pady=10, padx=5)
        
        save_btn = ttk.Button(buttons_frame, text="Salvar", 
                             command=lambda: self.save_slot_changes(slot_data))
        save_btn.pack(side='left', padx=(0, 5), fill='x', expand=True)
        
        cancel_btn = ttk.Button(buttons_frame, text="Cancelar", 
                               command=lambda: self.clear_right_panel())
        cancel_btn.pack(side='left', fill='x', expand=True)
        
        # Adiciona mais informações úteis
        info_frame = ttk.LabelFrame(self.right_panel, text="Informações")
        info_frame.pack(fill='x', pady=(10, 0), padx=5)
        
        # Tipo de slot
        tipo_frame = ttk.Frame(info_frame)
        tipo_frame.pack(fill='x', pady=2, padx=5)
        
        ttk.Label(tipo_frame, text="Tipo:", width=8).pack(side='left')
        tipo_value = ttk.Label(tipo_frame, text=slot_data.get('tipo', 'desconhecido'))
        tipo_value.pack(side='left', padx=(5, 0))
        
        # ID do slot
        id_frame = ttk.Frame(info_frame)
        id_frame.pack(fill='x', pady=2, padx=5)
        
        ttk.Label(id_frame, text="ID:", width=8).pack(side='left')
        id_value = ttk.Label(id_frame, text=str(slot_data['id']))
        id_value.pack(side='left', padx=(5, 0))
            


    
    def choose_color(self, color_key):
        """Abre o seletor de cores e atualiza o campo correspondente"""
        try:
            # Obtém a cor atual
            current_color = self.edit_vars[color_key].get()
            
            # Abre o seletor de cores
            color = colorchooser.askcolor(initialcolor=current_color, title=f"Escolher cor para {color_key}")
            
            # Se o usuário selecionou uma cor (não cancelou)
            if color and color[1]:
                # Atualiza o campo com a cor selecionada (formato hexadecimal)
                self.edit_vars[color_key].set(color[1])
                
                # Atualiza a interface para refletir a nova cor
                if color_key == "background_color":
                    self.edit_menu_frame.configure(bg=color[1])
                    for widget in self.edit_menu_frame.winfo_children():
                        if isinstance(widget, Canvas):
                            widget.configure(bg=color[1])
        except Exception as e:
            print(f"Erro ao escolher cor: {e}")
            messagebox.showerror("Erro", f"Erro ao escolher cor: {str(e)}")
    
    def choose_font(self, font_key):
        """Abre um diálogo para escolher a fonte"""
        try:
            # Obtém a fonte atual
            current_font = self.edit_vars[font_key].get()
            
            # Lista de fontes disponíveis
            available_fonts = [
                "Arial", "Arial Black", "Calibri", "Cambria", "Comic Sans MS", 
                "Courier New", "Georgia", "Impact", "Tahoma", "Times New Roman", 
                "Trebuchet MS", "Verdana"
            ]
            
            # Lista de tamanhos de fonte
            font_sizes = ["8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24", "28", "32", "36", "48", "72"]
            
            # Lista de estilos de fonte
            font_styles = ["normal", "bold", "italic", "bold italic"]
            
            # Cria uma janela de diálogo para escolher a fonte
            font_dialog = Toplevel(self.edit_menu_frame)
            font_dialog.title(f"Escolher fonte para {font_key}")
            font_dialog.geometry("400x300")
            font_dialog.transient(self.edit_menu_frame)
            font_dialog.grab_set()
            
            # Centraliza a janela
            font_dialog.update_idletasks()
            x = (font_dialog.winfo_screenwidth() // 2) - (200)
            y = (font_dialog.winfo_screenheight() // 2) - (150)
            font_dialog.geometry(f"400x300+{x}+{y}")
            
            # Frame principal
            main_frame = ttk.Frame(font_dialog, padding=10)
            main_frame.pack(fill='both', expand=True)
            
            # Variáveis para armazenar a seleção
            font_family_var = ttk.StringVar()
            font_size_var = ttk.StringVar()
            font_style_var = ttk.StringVar()
            
            # Tenta extrair os componentes da fonte atual
            try:
                # Formato esperado: "família tamanho estilo"
                font_parts = current_font.split()
                if len(font_parts) >= 3:
                    font_family_var.set(font_parts[0])
                    font_size_var.set(font_parts[1])
                    font_style_var.set(" ".join(font_parts[2:]))
                else:
                    # Valores padrão se não conseguir extrair
                    font_family_var.set("Arial")
                    font_size_var.set("12")
                    font_style_var.set("bold")
            except:
                # Valores padrão em caso de erro
                font_family_var.set("Arial")
                font_size_var.set("12")
                font_style_var.set("bold")
            
            # Frame para família da fonte
            family_frame = ttk.Frame(main_frame)
            family_frame.pack(fill='x', pady=5)
            
            ttk.Label(family_frame, text="Família:").pack(side=LEFT)
            family_combo = ttk.Combobox(family_frame, textvariable=font_family_var, values=available_fonts, width=20)
            family_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para tamanho da fonte
            size_frame = ttk.Frame(main_frame)
            size_frame.pack(fill='x', pady=5)
            
            ttk.Label(size_frame, text="Tamanho:").pack(side=LEFT)
            size_combo = ttk.Combobox(size_frame, textvariable=font_size_var, values=font_sizes, width=10)
            size_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para estilo da fonte
            style_frame = ttk.Frame(main_frame)
            style_frame.pack(fill='x', pady=5)
            
            ttk.Label(style_frame, text="Estilo:").pack(side=LEFT)
            style_combo = ttk.Combobox(style_frame, textvariable=font_style_var, values=font_styles, width=15)
            style_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para visualização
            preview_frame = ttk.Frame(main_frame, height=100)
            preview_frame.pack(fill='x', pady=10)
            preview_frame.pack_propagate(False)
            
            preview_label = ttk.Label(preview_frame, text="Texto de exemplo AaBbCcDd 123")
            preview_label.pack(expand=True)
            
            # Função para atualizar a visualização
            def update_preview(*args):
                try:
                    font_family = font_family_var.get()
                    font_size = int(font_size_var.get())
                    font_style = font_style_var.get()
                    
                    # Configura a fonte para o preview
                    preview_font = (font_family, font_size, font_style)
                    preview_label.configure(font=preview_font)
                except Exception as e:
                    print(f"Erro ao atualizar preview: {e}")
            
            # Vincula as variáveis à função de atualização
            font_family_var.trace_add("write", update_preview)
            font_size_var.trace_add("write", update_preview)
            font_style_var.trace_add("write", update_preview)
            
            # Atualiza o preview inicialmente
            update_preview()
            
            # Frame para botões
            buttons_frame = ttk.Frame(main_frame)
            buttons_frame.pack(fill='x', pady=10)
            
            # Função para aplicar a fonte selecionada
            def apply_font():
                try:
                    font_family = font_family_var.get()
                    font_size = font_size_var.get()
                    font_style = font_style_var.get()
                    
                    # Formata a string da fonte
                    font_string = f"{font_family} {font_size} {font_style}"
                    
                    # Atualiza a variável
                    self.edit_vars[font_key].set(font_string)
                    
                    # Fecha o diálogo
                    font_dialog.destroy()
                except Exception as e:
                    print(f"Erro ao aplicar fonte: {e}")
                    messagebox.showerror("Erro", f"Erro ao aplicar fonte: {str(e)}")
            
            # Botão OK
            ttk.Button(buttons_frame, text="OK", command=apply_font).pack(side=LEFT, padx=(0, 5))
            
            # Botão Cancelar
            ttk.Button(buttons_frame, text="Cancelar", command=font_dialog.destroy).pack(side=LEFT)
            
            # Torna a janela modal
            font_dialog.wait_window()
            
        except Exception as e:
            print(f"Erro ao escolher fonte: {e}")
            messagebox.showerror("Erro", f"Erro ao escolher fonte: {str(e)}")
    
    def save_inline_edit(self, slot_data):
        """Salva as alterações do menu inline"""
        try:
            # Atualiza os dados básicos do slot
            slot_data['x'] = int(self.edit_vars['x'].get())
            slot_data['y'] = int(self.edit_vars['y'].get())
            slot_data['w'] = int(self.edit_vars['w'].get())
            slot_data['h'] = int(self.edit_vars['h'].get())
            
            # Atualiza os parâmetros específicos para clips
            if slot_data.get('tipo') == 'clip':
                if 'detection_method' in self.edit_vars:
                    slot_data['detection_method'] = self.edit_vars['detection_method'].get()
                if 'detection_threshold' in self.edit_vars:
                    slot_data['detection_threshold'] = float(self.edit_vars['detection_threshold'].get())
                if 'correlation_threshold' in self.edit_vars:
                    slot_data['correlation_threshold'] = float(self.edit_vars['correlation_threshold'].get())
                if 'template_method' in self.edit_vars:
                    slot_data['template_method'] = self.edit_vars['template_method'].get()
                if 'scale_tolerance' in self.edit_vars:
                    slot_data['scale_tolerance'] = float(self.edit_vars['scale_tolerance'].get())
            
            # Salva no banco de dados se há um modelo carregado
            if self.current_model_id is not None:
                try:
                    self.db_manager.update_slot(slot_data['db_id'], slot_data)
                except Exception as e:
                    print(f"Erro ao salvar slot no banco: {e}")
            
            # Nota: Removido o processamento de configurações de estilo
            # Essas configurações agora são gerenciadas apenas pelo menu de configurações do sistema
            
            # Atualiza a exibição
            self.redraw_slots()
            self.update_slots_list()
            
            # Remove o menu de edição
            self.cancel_inline_edit()
            
            # Marca modelo como modificado
            self.mark_model_modified()
            
            print(f"Slot {slot_data['id']} atualizado com sucesso")
            messagebox.showinfo("Sucesso", f"Slot {slot_data['id']} foi atualizado com sucesso!")
            
        except ValueError as e:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            print(f"Erro ao salvar: {e}")
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
    def cancel_inline_edit(self):
        """Cancela a edição inline"""
        if hasattr(self, 'edit_menu_frame') and self.edit_menu_frame:
            self.edit_menu_frame.destroy()
            self.edit_menu_frame = None
        if hasattr(self, 'edit_vars'):
            self.edit_vars = None
    
    def update_slot_data(self, updated_slot_data):
        """Atualiza os dados de um slot específico."""
        slot_id_to_update = updated_slot_data.get('id')
        if slot_id_to_update is None:
            print("ERRO: ID do slot não encontrado nos dados atualizados")
            return
        
        print(f"\n=== ATUALIZANDO SLOT {slot_id_to_update} NA LISTA ===")
        print(f"Dados recebidos: {updated_slot_data}")
        
        found = False
        for i, slot in enumerate(self.slots):
            if slot['id'] == slot_id_to_update:
                print(f"Slot encontrado na posição {i}")
                print(f"Dados antigos: {slot}")
                
                # Preserva canvas_id se existir
                updated_slot_data['canvas_id'] = slot.get('canvas_id')
                
                # Substitui o slot na lista
                self.slots[i] = updated_slot_data
                found = True
                
                print(f"Dados novos: {self.slots[i]}")
                print(f"Slot {slot_id_to_update} atualizado com sucesso na lista.")
                break
        
        if not found:
            print(f"ERRO: Slot {slot_id_to_update} não encontrado na lista para update.")
            print(f"Slots disponíveis: {[s.get('id') for s in self.slots]}")
            return
        
        # Salva no banco de dados se há um modelo carregado
        if self.current_model_id is not None:
            try:
                print(f"Salvando slot {slot_id_to_update} no banco de dados...")
                self.db_manager.update_slot(self.current_model_id, updated_slot_data)
                print(f"Slot {slot_id_to_update} salvo no banco com sucesso!")
            except Exception as e:
                print(f"Erro ao salvar slot no banco: {e}")
        else:
            print("Aviso: Modelo não foi salvo ainda, dados atualizados apenas na memória.")
        
        print("Atualizando interface...")
        self.deselect_all_slots()
        self.redraw_slots()
        self.update_slots_list()
        
        # Marca o modelo como modificado
        self.mark_model_modified()
        
        print("Interface atualizada com sucesso!")    
    def delete_selected_slot(self):
        """Remove o slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Selecione um slot para deletar.")
            return
        
        if messagebox.askyesno("Confirmar", f"Deletar slot {self.selected_slot_id}?"):
            # Encontra o slot a ser removido
            slot_to_remove = None
            for slot in self.slots:
                if slot['id'] == self.selected_slot_id:
                    slot_to_remove = slot
                    break
            
            # Remove do banco de dados se há um modelo carregado
            if self.current_model_id is not None and slot_to_remove and 'db_id' in slot_to_remove:
                try:
                    self.db_manager.delete_slot(slot_to_remove['db_id'])
                except Exception as e:
                    print(f"Erro ao remover slot do banco: {e}")
            
            # Remove slot da lista
            self.slots = [slot for slot in self.slots if slot['id'] != self.selected_slot_id]
            
            # Remove seleção
            self.selected_slot_id = None
            
            # Atualiza interface
            self.update_slots_list()
            self.redraw_slots()
            self.status_var.set("Slot deletado")
            self.update_button_states()
            
            # Marca modelo como modificado
            self.mark_model_modified()
    
    def train_selected_slot(self):
        """Abre o diálogo de treinamento para o slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Nenhum slot selecionado.")
            return
        
        # Encontra o slot
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if selected_slot is None:
            messagebox.showerror("Erro", "Slot não encontrado.")
            return
        
        if selected_slot.get('tipo') != 'clip':
            messagebox.showwarning("Aviso", "Treinamento disponível apenas para slots do tipo 'clip'.")
            return
        
        # Abre diálogo de treinamento
        dialog = SlotTrainingDialog(self.master, selected_slot, self)
        dialog.wait_window()
        
        # Atualiza interface após treinamento
        self.redraw_slots()
        self.update_slots_list()
    
    def save_templates_to_model_folder(self, model_name, model_id):
        """Salva todos os templates dos slots na pasta do modelo."""
        try:
            # Obtém pasta de templates do modelo
            model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
            templates_folder = model_folder / "templates"
            templates_folder.mkdir(parents=True, exist_ok=True)
            
            # Salva cada template dos slots
            for slot_data in self.slots:
                if 'roi_data' in slot_data and 'template_filename' in slot_data:
                    template_path = templates_folder / slot_data['template_filename']
                    cv2.imwrite(str(template_path), slot_data['roi_data'])
                    
                    # Atualiza o caminho do template no slot
                    slot_data['template_path'] = str(template_path)
                    
                    # Remove os dados temporários
                    del slot_data['roi_data']
                    del slot_data['template_filename']
                    
                    print(f"Template salvo: {template_path}")
                    
        except Exception as e:
            print(f"Erro ao salvar templates: {e}")
            raise e
    
    def save_model(self):
        """Salva o modelo atual no banco de dados."""
        if self.img_original is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return
        
        if not self.slots:
            messagebox.showwarning("Aviso", "Nenhum slot definido para salvar.")
            return
        
        # Abre diálogo para salvar modelo
        dialog = SaveModelDialog(self, self.db_manager, self.current_model_id)
        result = dialog.show()
        
        if not result:
            return
        
        try:
            # Determina o nome do modelo
            if 'name' in result:
                model_name = result['name']
            elif result['action'] == 'overwrite' and 'model_id' in result:
                # Para sobrescrever, busca o nome do modelo existente
                existing_model = self.db_manager.load_modelo(result['model_id'])
                model_name = existing_model['nome']
            else:
                raise ValueError("Nome do modelo não encontrado")
            
            if result['action'] in ['update', 'overwrite']:
                # Atualiza modelo existente
                model_id = result['model_id']
                
                # Salva templates primeiro
                self.save_templates_to_model_folder(model_name, model_id)
                
                # Obtém pasta específica do modelo
                model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
                
                # Salva imagem de referência na pasta do modelo
                image_filename = f"{model_name}_reference.jpg"
                image_path = model_folder / image_filename
                cv2.imwrite(str(image_path), self.img_original)
                
                self.db_manager.update_modelo(
                    model_id,
                    nome=model_name,
                    image_path=str(image_path),
                    slots=self.slots
                )
                
                self.current_model_id = model_id
                # Define o modelo atual para uso em outras funções
                self.current_model = self.db_manager.load_modelo(model_id)
                
            else:
                # Cria novo modelo primeiro para obter o ID
                # Salva temporariamente com caminho vazio
                model_id = self.db_manager.save_modelo(
                    nome=model_name,
                    image_path="",  # Será atualizado depois
                    slots=[]
                )
                
                # Agora salva os templates na pasta correta do modelo
                self.save_templates_to_model_folder(model_name, model_id)
                
                # Obtém pasta específica do modelo (já criada pelo save_modelo)
                model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
                
                # Salva imagem de referência na pasta do modelo
                image_filename = f"{model_name}_reference.jpg"
                image_path = model_folder / image_filename
                cv2.imwrite(str(image_path), self.img_original)
                
                # Atualiza o modelo com os slots e caminho da imagem
                self.db_manager.update_modelo(
                    model_id,
                    image_path=str(image_path),
                    slots=self.slots
                )
                
                self.current_model_id = model_id
                # Define o modelo atual para uso em outras funções
                self.current_model = self.db_manager.load_modelo(model_id)
            
            # Marca o modelo como salvo
            self.mark_model_saved()
            
            print(f"Modelo '{model_name}' salvo com sucesso no banco de dados")
            messagebox.showinfo("Sucesso", f"Modelo '{model_name}' salvo com {len(self.slots)} slots.")
            
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao salvar modelo: {str(e)}")
    
    def update_button_states(self):
        """Atualiza estado dos botões baseado no estado atual."""
        has_image = self.img_original is not None
        has_slots = len(self.slots) > 0
        has_selection = self.selected_slot_id is not None
        
        # Botões que dependem de imagem
        if hasattr(self, 'btn_save_model'):
            self.btn_save_model.config(state=NORMAL if has_image and has_slots else DISABLED)
        
        # Botões que dependem de slots
        if hasattr(self, 'btn_clear_slots'):
            self.btn_clear_slots.config(state=NORMAL if has_slots else DISABLED)
        
        # Botões que dependem de seleção
        if hasattr(self, 'btn_delete_slot'):
            self.btn_delete_slot.config(state=NORMAL if has_selection else DISABLED)
        if hasattr(self, 'btn_train_slot'):
            self.btn_train_slot.config(state=NORMAL if has_selection else DISABLED)
    
    def set_drawing_mode(self):
        """Define o modo de desenho atual."""
        self.current_drawing_mode = self.drawing_mode.get()
        mode_names = {
            "rectangle": "Retângulo",
            "exclusion": "Área de Exclusão"
        }
        self.tool_status_var.set(f"Modo: {mode_names.get(self.current_drawing_mode, 'Desconhecido')}")
        print(f"Modo de desenho alterado para: {self.current_drawing_mode}")
    
    # Função de rotação removida
    
    def add_exclusion_area(self, x, y, w, h):
        """Adiciona área de exclusão ao slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Selecione um slot primeiro para adicionar área de exclusão.")
            return
        
        # Encontra o slot selecionado
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if selected_slot is None:
            messagebox.showerror("Erro", "Slot selecionado não encontrado.")
            return
        
        # Adiciona área de exclusão (sem verificação de limites)
        exclusion_area = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'shape': self.current_drawing_mode
        }
        
        selected_slot['exclusion_areas'].append(exclusion_area)
        self.mark_model_modified()
        self.redraw_slots()
        
        print(f"Área de exclusão adicionada ao slot {self.selected_slot_id}: ({x}, {y}, {w}, {h})")
        self.status_var.set(f"Área de exclusão adicionada ao slot {self.selected_slot_id}")
    
    def show_edit_handles(self, slot):
        """Mostra handles de edição para o slot selecionado."""
        self.hide_edit_handles()  # Remove handles anteriores
        
        x = slot['x'] * self.scale_factor
        y = slot['y'] * self.scale_factor
        w = slot['w'] * self.scale_factor
        h = slot['h'] * self.scale_factor
        
        handle_size = 8
        handle_color = "#FF4444"
        
        # Handles de redimensionamento (cantos e meio das bordas)
        handles = [
            # Cantos
            (x - handle_size//2, y - handle_size//2, "nw"),  # Canto superior esquerdo
            (x + w - handle_size//2, y - handle_size//2, "ne"),  # Canto superior direito
            (x - handle_size//2, y + h - handle_size//2, "sw"),  # Canto inferior esquerdo
            (x + w - handle_size//2, y + h - handle_size//2, "se"),  # Canto inferior direito
            # Meio das bordas
            (x + w//2 - handle_size//2, y - handle_size//2, "n"),  # Meio superior
            (x + w//2 - handle_size//2, y + h - handle_size//2, "s"),  # Meio inferior
            (x - handle_size//2, y + h//2 - handle_size//2, "w"),  # Meio esquerdo
            (x + w - handle_size//2, y + h//2 - handle_size//2, "e"),  # Meio direito
        ]
        
        # Handle de rotação removido
        
        # Cria handles de redimensionamento
        for hx, hy, direction in handles:
            handle = self.canvas.create_rectangle(
                hx, hy, hx + handle_size, hy + handle_size,
                fill=handle_color, outline="white", width=2,
                tags=("edit_handle", f"resize_handle_{direction}")
            )
        
        # Bind eventos para os handles
        self.canvas.tag_bind("edit_handle", "<Button-1>", self.on_handle_press)
        self.canvas.tag_bind("edit_handle", "<B1-Motion>", self.on_handle_drag)
        self.canvas.tag_bind("edit_handle", "<ButtonRelease-1>", self.on_handle_release)
    
    def hide_edit_handles(self):
        """Esconde todos os handles de edição."""
        self.canvas.delete("edit_handle")
        self.editing_handle = None
    
    def on_handle_press(self, event):
        """Inicia edição com handle."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Encontra qual handle foi clicado
        closest_items = self.canvas.find_closest(canvas_x, canvas_y)
        if closest_items:
            item = closest_items[0]
            tags = self.canvas.gettags(item)
            
            for tag in tags:
                if tag.startswith("resize_handle_"):
                    self.editing_handle = {
                        'type': 'resize',
                        'direction': tag.split('_')[-1],
                        'start_x': canvas_x,
                        'start_y': canvas_y
                    }
                    break
                # Tratamento de handle de rotação removido
    
    def on_handle_drag(self, event):
        """Processa arrastar do handle."""
        if not self.editing_handle or self.selected_slot_id is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Encontra o slot selecionado
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if not selected_slot:
            return
        
        if self.editing_handle['type'] == 'resize':
            self.handle_resize_drag(selected_slot, canvas_x, canvas_y)
        # Tratamento de arrastar handle de rotação removido
    
    def handle_resize_drag(self, slot, canvas_x, canvas_y):
        """Lida com redimensionamento do slot."""
        direction = self.editing_handle['direction']
        
        # Converte coordenadas do canvas para coordenadas da imagem
        img_x = canvas_x / self.scale_factor
        img_y = canvas_y / self.scale_factor
        
        # Calcula novas dimensões baseadas na direção do handle
        new_x, new_y = slot['x'], slot['y']
        new_w, new_h = slot['w'], slot['h']
        
        if 'w' in direction:  # Lado esquerdo
            new_w = slot['x'] + slot['w'] - img_x
            new_x = img_x
        elif 'e' in direction:  # Lado direito
            new_w = img_x - slot['x']
        
        if 'n' in direction:  # Lado superior
            new_h = slot['y'] + slot['h'] - img_y
            new_y = img_y
        elif 's' in direction:  # Lado inferior
            new_h = img_y - slot['y']
        
        # Garante dimensões mínimas
        if new_w < 10:
            new_w = 10
        if new_h < 10:
            new_h = 10
        
        # Atualiza o slot
        slot['x'] = max(0, new_x)
        slot['y'] = max(0, new_y)
        slot['w'] = new_w
        slot['h'] = new_h
        
        # Marca modelo como modificado e atualiza interface
        self.mark_model_modified()
        self.redraw_slots()
        self.show_edit_handles(slot)
        self.update_slots_list()
    
    # Função de rotação removida
    
    def on_handle_release(self, event):
        """Finaliza edição com handle."""
        if self.editing_handle:
            self.mark_model_modified()
            self.editing_handle = None
     
    def show_help(self):
        """Mostra janela de ajuda."""
        help_window = Toplevel(self.master)
        help_window.title("Ajuda - Editor de Malha")
        help_window.geometry("600x500")
        help_window.resizable(True, True)
        
        # Torna a janela modal
        help_window.transient(self.master)
        help_window.grab_set()
        
        # Texto de ajuda

    def validate_slot_reference(self, slot_id):
        """Valida se a referência do slot está correta para o modelo atual."""
        try:
            current_model = getattr(self, 'current_model', None)
            if not current_model:
                return False
            
            if slot_id not in self.slots:
                return False
            
            slot_data = self.slots[slot_id]
            template_path = slot_data.get('template_path')
            
            if template_path:
                # Verifica se o template pertence ao modelo atual
                expected_prefix = f"modelos/{current_model['nome']}_{current_model['id']}/templates/"
                if not template_path.startswith(expected_prefix):
                    print(f"⚠️ Template {template_path} não pertence ao modelo {current_model['nome']}")
                    return False
                
                # Verifica se o arquivo existe
                abs_path = get_project_root() / template_path
                if not abs_path.exists():
                    print(f"⚠️ Template não encontrado: {template_path}")
                    return False
            
            return True
        except Exception as e:
            print(f"❌ Erro na validação do slot {slot_id}: {e}")
            return False
    
    def cleanup_orphaned_templates(self):
        """Remove templates órfãos que não pertencem ao modelo atual."""
        try:
            current_model = getattr(self, 'current_model', None)
            if not current_model:
                return
            
            orphaned_count = 0
            for slot_id, slot_data in self.slots.items():
                if not self.validate_slot_reference(slot_id):
                    # Remove referência inválida
                    if 'template_path' in slot_data:
                        print(f"🧹 Removendo referência órfã do slot {slot_id}: {slot_data['template_path']}")
                        del slot_data['template_path']
                        orphaned_count += 1
            
            if orphaned_count > 0:
                print(f"✅ {orphaned_count} referências órfãs removidas")
                
        except Exception as e:
            print(f"❌ Erro na limpeza de templates órfãos: {e}")

        help_text = """
# Editor de Malha - Ajuda

## Como usar:

### 1. Carregar Imagem
- Clique em "Carregar Imagem" para selecionar uma imagem de referência
- Formatos suportados: JPG, PNG, BMP, TIFF

### 2. Criar Slots
- Clique e arraste no canvas para desenhar um retângulo
- Apenas slots do tipo 'clip' são suportados
- Será salvo um template da região para template matching

### 3. Gerenciar Slots
- Clique em um slot para selecioná-lo
- Use "Editar Slot" para modificar configurações
- Use "Deletar Slot" para remover um slot
- Use "Limpar Slots" para remover todos os slots

### 4. Salvar/Carregar Modelos
- Use "Salvar Modelo" para salvar a configuração atual
- Use "Carregar Modelo" para carregar uma configuração existente
- Os modelos são salvos em formato JSON

### 5. Cores dos Slots
- Clips: Vermelho coral
- Selecionado: Amarelo dourado
- Desenhando: Verde claro

## Dicas:
- Slots muito pequenos (< 10x10 pixels) não são aceitos
- Templates de clips são salvos automaticamente
- Use zoom e scroll para trabalhar com imagens grandes
- Modelos salvam caminhos relativos para portabilidade
"""
        
        # Frame principal
        main_frame = ttk.Frame(help_window)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Área de texto com scroll
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Texto
        text_widget = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set,
                          font=("Consolas", 10), bg="#F8F9FA", fg="#2C3E50")
        text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insere texto
        text_widget.insert("1.0", help_text)
        text_widget.config(state=DISABLED)
        
        # Botão fechar
        ttk.Button(main_frame, text="Fechar", 
                  command=help_window.destroy).pack(pady=(10, 0))
        
        # Centralizar janela
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (help_window.winfo_width() // 2)
        y = (help_window.winfo_screenheight() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")
    
    def open_system_config(self):
        """Abre a janela de configuração do sistema."""
        config_dialog = SystemConfigDialog(self.master)
        config_dialog.wait_window()
    
    def set_drawing_mode(self):
        """Define o modo de desenho atual."""
        mode = self.drawing_mode.get()
        if mode == "rectangle":
            self.tool_status_var.set("Modo: Retângulo")
        elif mode == "exclusion":
            self.tool_status_var.set("Modo: Exclusão")
        self.current_drawing_mode = mode
    
    # Funções de rotação removidas

    def on_closing(self):
        """Limpa recursos ao fechar a aplicação."""
        if self.live_capture:
            self.stop_live_capture()
        self.master.destroy()


# Classe para aba de Inspeção
class InspecaoWindow(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Dados da aplicação
        self.img_reference = None
        self.img_test = None
        self.img_display = None
        self.scale_factor = 1.0
        self.slots = []
        self.current_model_id = None
        self.inspection_results = []
        
        # Inicializa gerenciador de banco de dados
        # Usa caminho absoluto baseado na raiz do projeto
        db_path = MODEL_DIR / "models.db"
        self.db_manager = DatabaseManager(str(db_path))
        
        # Estado da inspeção
        self.live_view = False
        self.camera = None
        self.live_capture = False
        self.latest_frame = None
        
        # Controle de webcam
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        
        self.setup_ui()
        self.update_button_states()
        
        # Inicia câmera em segundo plano após inicialização completa
        if self.available_cameras:
            self.after(500, lambda: self.start_background_camera_direct(self.available_cameras[0]))
            
    def start_background_camera_direct(self, camera_index):
        """Inicia a câmera diretamente em segundo plano com índice específico."""
        try:
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução padrão para inicialização rápida
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            print(f"Webcam {camera_index} inicializada com sucesso em segundo plano")
            
            # Inicia captura de frames em thread separada
            self.start_background_frame_capture()
            
        except Exception as e:
            print(f"Erro ao inicializar webcam {camera_index}: {e}")
            self.camera = None
            self.live_capture = False
    
    def setup_ui(self):
        # Configuração de estilo industrial Keyence
        self.style = ttk.Style()
        
        # Carrega as configurações de estilo personalizadas
        style_config = load_style_config()
        
        # Cores industriais Keyence com personalização
        self.bg_color = style_config["background_color"]  # Fundo escuro mais profundo
        self.panel_color = "#2A2A2A"  # Cor dos painéis
        self.accent_color = style_config["button_color"]  # Cor de destaque
        self.success_color = style_config["ok_color"]  # Verde brilhante industrial
        self.warning_color = "#FFCC00"  # Amarelo industrial
        self.danger_color = style_config["ng_color"]  # Vermelho industrial
        self.text_color = style_config["text_color"]  # Texto branco
        self.button_bg = "#3A3A3A"  # Cor de fundo dos botões
        self.button_active = "#4A4A4A"  # Cor quando botão ativo
        
        # Configurar estilos
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TLabelframe', background=self.panel_color, borderwidth=2, relief='groove')
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent_color, 
                             font=style_config["ok_font"])
        
        # Botões com estilo industrial
        self.style.configure('TButton', background=self.button_bg, foreground=self.text_color, 
                             font=style_config["ok_font"], borderwidth=2, relief='raised')
        self.style.map('TButton', 
                       background=[('active', self.button_active), ('pressed', self.accent_color)],
                       foreground=[('pressed', 'white')])
        
        # Estilo para botão de inspeção (destaque)
        self.style.configure('Inspect.TButton', font=style_config["ok_font"], background=self.accent_color)
        self.style.map('Inspect.TButton',
                       background=[('active', '#FF7733'), ('pressed', '#CC4400')])
        
        # Estilos para resultados
        self.style.configure('Success.TFrame', background='#004400')
        self.style.configure('Danger.TFrame', background='#440000')
        
        # Estilos para Entry e Combobox
        self.style.configure('TEntry', fieldbackground='#2A2A2A', foreground=self.text_color)
        self.style.map('TEntry',
                       fieldbackground=[('readonly', '#2A2A2A')],
                       foreground=[('readonly', self.text_color)])
        
        self.style.configure('TCombobox', fieldbackground='#2A2A2A', foreground=self.text_color, selectbackground='#3A3A3A')
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', '#2A2A2A')],
                       foreground=[('readonly', self.text_color)])
        
        # Configurar cores para a interface - usando style em vez de configure diretamente
        # Nota: widgets ttk não suportam configuração direta de background
        # self.configure(background=self.bg_color) # Esta linha causava erro
        
        # Frame principal com layout horizontal de três painéis
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        # Painel central - Apenas imagem
        center_panel = ttk.Frame(main_frame)
        center_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Painel direito - Resultados e status
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=Y, padx=(0, 0), pady=0, ipadx=0)
        
        # === PAINEL ESQUERDO ===
        
        # Cabeçalho com título estilo Keyence
        header_frame = ttk.Frame(left_panel, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 15))
        
        # Estilo para o cabeçalho
        self.style.configure('Header.TFrame', background=self.accent_color)
        
        # Logo e título
        header_label = ttk.Label(header_frame, text="AutoVerify DX SYSTEM", 
                                font=style_config["ok_font"], foreground="white",
                                background=self.accent_color)
        header_label.pack(pady=10, fill=X)
        
        # Versão do sistema
        version_label = ttk.Label(header_frame, text="V1.0.0 - INDUSTRIAL INSPECTION", 
                                font=style_config["ok_font"].replace("12", "8"), foreground="white",
                                background=self.accent_color)
        version_label.pack(pady=(0, 10))
        
        # Seção de Modelo - Estilo industrial Keyence
        model_frame = ttk.LabelFrame(left_panel, text="MODELO DE INSPEÇÃO")
        model_frame.pack(fill=X, pady=(0, 10))
        
        # Indicador de modelo carregado
        model_indicator_frame = ttk.Frame(model_frame)
        model_indicator_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(model_indicator_frame, text="STATUS:", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(0, 5))
        
        self.model_status_var = StringVar(value="NÃO CARREGADO")
        model_status = ttk.Label(model_indicator_frame, textvariable=self.model_status_var, 
                                foreground=self.danger_color, font=("Arial", 8, "bold"))
        model_status.pack(side=LEFT)
        
        # Botão com ícone industrial
        self.btn_load_model = ttk.Button(model_frame, text="CARREGAR MODELO ▼", 
                                       command=self.load_model_dialog, style="Action.TButton")
        self.btn_load_model.pack(fill=X, padx=5, pady=5)
        
        # Seção de Imagem de Teste - Estilo industrial
        test_frame = ttk.LabelFrame(left_panel, text="IMAGEM DE TESTE")
        test_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_load_test = ttk.Button(test_frame, text="CARREGAR IMAGEM", 
                                       command=self.load_test_image)
        self.btn_load_test.pack(fill=X, padx=5, pady=2)
        
        # Seção de Webcam - Estilo industrial
        webcam_frame = ttk.LabelFrame(left_panel, text="CÂMERA")
        webcam_frame.pack(fill=X, pady=(0, 10))
        
        # Combobox para seleção de câmera
        camera_selection_frame = ttk.Frame(webcam_frame)
        camera_selection_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(camera_selection_frame, text="ID:").pack(side=LEFT)
        self.camera_combo = Combobox(camera_selection_frame, 
                                   values=[str(i) for i in self.available_cameras],
                                   state="readonly", width=5)
        self.camera_combo.pack(side=RIGHT)
        if self.available_cameras:
            self.camera_combo.set(str(self.available_cameras[0]))
        
        # Nota informativa sobre o ajuste automático
        info_frame = ttk.Frame(webcam_frame)
        info_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(info_frame, text="A imagem será ajustada automaticamente", 
                 font=("Arial", 7), foreground="#AAAAAA")\
            .pack(side=LEFT, padx=(0, 5))
        
        # Botão para iniciar/parar captura contínua
        self.btn_capture_test = ttk.Button(webcam_frame, text="CAPTURAR IMAGEM", 
                                          command=self.capture_test_from_webcam)
        self.btn_capture_test.pack(fill=X, padx=5, pady=2)
        
        # Seção de Inspeção - Estilo industrial Keyence com destaque
        inspection_frame = ttk.LabelFrame(left_panel, text="INSPEÇÃO AUTOMÁTICA")
        inspection_frame.pack(fill=X, pady=(0, 10))
        
        # Indicador de status de inspeção
        inspection_status_frame = ttk.Frame(inspection_frame)
        inspection_status_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(inspection_status_frame, text="SISTEMA:", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(0, 5))
        
        self.inspection_status_var = StringVar(value="PRONTO")
        self.inspection_status_label = ttk.Label(inspection_status_frame, textvariable=self.inspection_status_var, 
                                     foreground=self.success_color, font=("Arial", 8, "bold"))
        self.inspection_status_label.pack(side=LEFT)
        
        # Botões de inspeção contínua removidos conforme solicitado pelo usuário
        
        # Botão para inspecionar sem tirar foto
        self.btn_inspect_only = ttk.Button(inspection_frame, text="INSPECIONAR SEM CAPTURAR", 
                                        command=self.inspect_without_capture,
                                        style="Inspect.TButton")
        self.btn_inspect_only.pack(fill=X, padx=5, pady=5)
        
        # === PAINEL CENTRAL - CANVAS DE INSPEÇÃO ===
        
        # Canvas de inspeção com estilo industrial - Ocupando toda a área central
        canvas_frame = ttk.LabelFrame(center_panel, text="VISUALIZAÇÃO DE INSPEÇÃO")
        canvas_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # Frame para canvas e scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        # Canvas com fundo escuro estilo industrial - Ampliado para ocupar toda a área
        self.canvas = Canvas(canvas_container, bg="#121212",
                           yscrollcommand=v_scrollbar.set,
                           xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Configurar scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Adicionar evento de redimensionamento para ajustar a imagem
        def on_canvas_configure(event):
            # Atualiza a exibição quando o canvas é redimensionado
            if hasattr(self, 'img_test') and self.img_test is not None:
                self.update_display()
        
        # Vincular evento de configuração (redimensionamento) ao canvas
        self.canvas.bind('<Configure>', on_canvas_configure)
        
        # === PAINEL DIREITO - STATUS E RESULTADOS ===
        
        # Reorganização: Painel de status expandido no topo
        status_summary_frame = ttk.LabelFrame(right_panel, text="PAINEL DE STATUS")
        status_summary_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Frame interno para o grid de status
        self.status_grid_frame = ttk.Frame(status_summary_frame)
        self.status_grid_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Resultados - Estilo industrial Keyence (reduzido) - Movido para parte inferior
        results_frame = ttk.LabelFrame(right_panel, text="RESULTADOS DE INSPEÇÃO")
        results_frame.pack(fill=X, expand=False, side=BOTTOM, pady=(0, 10))
        
        # Painel de resumo de resultados
        summary_frame = ttk.Frame(results_frame)
        summary_frame.pack(fill=X, padx=5, pady=5)
        
        # Criar painel de resumo de status
        self.create_status_summary_panel(summary_frame)
        
        # Lista de resultados com estilo industrial (altura reduzida)
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=X, expand=False, padx=5, pady=5)
        
        scrollbar_results = ttk.Scrollbar(list_container)
        scrollbar_results.pack(side=RIGHT, fill=Y)
        
        # Configurar estilo da Treeview para parecer com sistemas Keyence
        self.style.configure("Treeview", 
                           background="#222222", 
                           foreground=self.text_color, 
                           fieldbackground="#222222",
                           borderwidth=1,
                           relief="solid")
        self.style.configure("Treeview.Heading", 
                           font=style_config["ok_font"], 
                           background="#444444", 
                           foreground="#FFFFFF")
        self.style.map("Treeview", 
                      background=[("selected", style_config["selection_color"])],
                      foreground=[("selected", "#000000")])
        
        # Altura reduzida para 4 linhas em vez de 8
        self.results_listbox = ttk.Treeview(list_container, yscrollcommand=scrollbar_results.set, height=4)
        self.results_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar_results.config(command=self.results_listbox.yview)
        
        # Configurar colunas da lista de resultados
        self.results_listbox["columns"] = ("status", "score", "detalhes")
        self.results_listbox.column("#0", width=40, minwidth=40)
        self.results_listbox.column("status", width=60, minwidth=60, anchor="center")
        self.results_listbox.column("score", width=60, minwidth=60, anchor="center")
        self.results_listbox.column("detalhes", width=120, minwidth=120)
        
        self.results_listbox.heading("#0", text="SLOT")
        self.results_listbox.heading("status", text="STATUS")
        self.results_listbox.heading("score", text="SCORE")
        self.results_listbox.heading("detalhes", text="DETALHES")
        
        # Configurar tags para resultados
        self.results_listbox.tag_configure("pass", background="#004400", foreground="#FFFFFF")
        self.results_listbox.tag_configure("fail", background="#440000", foreground="#FFFFFF")
        
        # Dicionário para armazenar widgets de status
        self.status_widgets = {}
        
        # Status bar estilo industrial
        status_bar_frame = ttk.Frame(self)
        status_bar_frame.pack(side=BOTTOM, fill=X)
        
        self.status_var = StringVar()
        self.status_var.set("SISTEMA PRONTO - CARREGUE UM MODELO PARA COMEÇAR")
        
        # Armazenar referência ao status_bar para poder modificar suas propriedades
        self.status_bar = ttk.Label(status_bar_frame, textvariable=self.status_var, 
                                  relief="sunken", font=style_config["ok_font"].replace("12", "9"))
        self.status_bar.pack(side=LEFT, fill=X, expand=True, padx=2, pady=2)
    
    def load_model_dialog(self):
        """Abre diálogo para selecionar modelo do banco de dados."""
        dialog = ModelSelectorDialog(self, self.db_manager)
        result = dialog.show()
        
        if result:
            if result['action'] == 'load':
                self.load_model_from_db(result['model_id'])
    
    def load_model_from_db(self, model_id):
        """Carrega um modelo do banco de dados."""
        try:
            # Carrega dados do modelo
            model_data = self.db_manager.load_modelo(model_id)
            
            # Carrega imagem de referência
            image_path = model_data['image_path']
            
            # Tenta caminho absoluto primeiro
            if not Path(image_path).exists():
                # Tenta caminho relativo ao diretório de modelos
                relative_path = MODEL_DIR / Path(image_path).name
                if relative_path.exists():
                    image_path = str(relative_path)
                else:
                    raise FileNotFoundError(f"Imagem de referência não encontrada: {image_path}")
            
            self.img_reference = cv2.imread(str(image_path))
            if self.img_reference is None:
                raise ValueError(f"Não foi possível carregar a imagem de referência: {image_path}")
            
            # Carrega slots
            self.slots = model_data['slots']
            self.current_model_id = model_id
            # Define o modelo atual para uso em outras funções
            self.current_model = model_data
            
            # Limpa resultados de inspeção anteriores
            self.inspection_results = []
            
            # Limpa a lista de resultados na interface
            children = self.results_listbox.get_children()
            if children:
                self.results_listbox.delete(*children)
            
            # Criar painel de resumo de status
            self.create_status_summary_panel()
            
            self.status_var.set(f"Modelo carregado: {model_data['nome']} ({len(self.slots)} slots)")
            self.update_button_states()
            
            print(f"Modelo de inspeção '{model_data['nome']}' carregado com sucesso: {len(self.slots)} slots")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.status_var.set(f"Erro ao carregar modelo: {str(e)}")
    
    def load_test_image(self):
        """Carrega imagem de teste."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem de Teste",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.img_test = cv2.imread(str(file_path))
                if self.img_test is None:
                    raise ValueError(f"Não foi possível carregar a imagem: {file_path}")
                

                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                self.update_display()
                self.status_var.set(f"Imagem de teste carregada: {Path(file_path).name}")
                self.update_button_states()
                
            except Exception as e:
                print(f"Erro ao carregar imagem de teste: {e}")
                self.status_var.set(f"Erro ao carregar imagem: {str(e)}")
    
    def start_live_capture_inspection(self):
        """Inicia captura contínua da câmera em segundo plano para inspeção automática."""
        # Verifica se o atributo live_capture existe
        if not hasattr(self, 'live_capture'):
            self.live_capture = False
            
        if self.live_capture:
            return
            
        try:
            # Desativa o modo de inspeção manual se estiver ativo
            if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
                try:
                    self.stop_live_capture_manual_inspection()
                except Exception as stop_error:
                    print(f"Erro ao parar inspeção manual: {stop_error}")
                
            # Verifica se o atributo camera_combo existe
            if not hasattr(self, 'camera_combo'):
                raise ValueError("Seletor de câmera não encontrado")
                
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Para live view se estiver ativo
            if self.live_view:
                try:
                    self.stop_live_view()
                except Exception as stop_view_error:
                    print(f"Erro ao parar visualização ao vivo: {stop_view_error}")
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance e inicialização rápida
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Inicializa contador de frames para inspeção automática
            self._inspection_frame_count = 0
            
            self.live_capture = True
            self.manual_inspection_mode = False  # Garante que o modo de inspeção manual está desativado
            
            # Inicia o processamento de frames
            try:
                self.process_live_frame_inspection()
            except Exception as process_error:
                print(f"Erro ao iniciar processamento de frames: {process_error}")
                
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Inspeção com Enter iniciada - Câmera {camera_index} ativa - Pressione ENTER para inspecionar")
            
            # Limpa resultados anteriores
            self.inspection_results = []
            self.update_results_list()
            
            # Configura o bind da tecla Enter para inspeção
            if hasattr(self, 'master'):
                try:
                    self.master.bind('<Return>', self.on_enter_key_continuous_inspection)
                except Exception as bind_error:
                    print(f"Erro ao configurar tecla Enter para inspeção contínua: {bind_error}")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera para inspeção contínua: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera para inspeção contínua: {str(e)}")
    
    def stop_live_capture_inspection(self):
        """Para a captura contínua da câmera para inspeção."""
        try:
            # Verifica se os atributos existem antes de acessá-los
            if hasattr(self, 'live_capture'):
                self.live_capture = False
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Libera a câmera se existir e não estiver sendo usada pelo live_view
            if hasattr(self, 'camera') and self.camera is not None and not self.live_view:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera: {release_error}")
            
            # Remove o bind da tecla Enter
            if hasattr(self, 'master'):
                try:
                    self.master.unbind('<Return>')
                except Exception as unbind_error:
                    print(f"Erro ao remover bind da tecla Enter: {unbind_error}")
            
            # Limpa o frame mais recente
            if hasattr(self, 'latest_frame'):
                self.latest_frame = None
                
            # Reseta o contador de frames de inspeção
            if hasattr(self, '_inspection_frame_count'):
                self._inspection_frame_count = 0
                
            # Atualiza o status
            if hasattr(self, 'live_view') and not self.live_view and hasattr(self, 'status_var'):
                self.status_var.set("Câmera desconectada")
                
        except Exception as e:
            print(f"Erro ao parar captura para inspeção contínua: {e}")
            # Não exibe messagebox para evitar interrupção da interface
            
    def start_live_capture_manual_inspection(self):
        """Inicia captura contínua da câmera em segundo plano para inspeção manual com Enter."""
        # Verifica se o atributo live_capture existe
        if not hasattr(self, 'live_capture'):
            self.live_capture = False
            
        if self.live_capture:
            return
        
        try:
            # Verifica se há um modelo carregado para inspeção
            if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                # Apenas atualiza o status
                if hasattr(self, 'status_var'):
                    self.status_var.set("É necessário carregar um modelo de inspeção para iniciar a captura")
                # Referência ao botão removido - btn_continuous_inspect
                return
            
            # Verifica se o atributo camera_combo existe
            if not hasattr(self, 'camera_combo'):
                raise ValueError("Seletor de câmera não encontrado")
                
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Para live view se estiver ativo
            if self.live_view:
                try:
                    self.stop_live_view()
                except Exception as stop_view_error:
                    print(f"Erro ao parar visualização ao vivo: {stop_view_error}")
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance e inicialização rápida
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            self.manual_inspection_mode = True  # Modo de inspeção manual
            
            # Inicia o processamento de frames
            try:
                self.process_live_frame_manual_inspection()
            except Exception as process_error:
                print(f"Erro ao iniciar processamento de frames para inspeção manual: {process_error}")
                
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Câmera {camera_index} ativa - Pressione ENTER para capturar e inspecionar")
            
            # Referência ao botão removido - btn_continuous_inspect
            
            # Limpa resultados anteriores
            if hasattr(self, 'inspection_results'):
                self.inspection_results = []
                
            # Atualiza a lista de resultados
            if hasattr(self, 'update_results_list'):
                try:
                    self.update_results_list()
                except Exception as update_error:
                    print(f"Erro ao atualizar lista de resultados: {update_error}")
            
            # Configura o bind da tecla Enter para inspeção
            if hasattr(self, 'master'):
                try:
                    self.master.bind('<Return>', self.on_enter_key_inspection)
                except Exception as bind_error:
                    print(f"Erro ao configurar tecla Enter para inspeção: {bind_error}")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera para inspeção manual: {e}")
            # Não exibe messagebox quando chamado automaticamente ao entrar na aba
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Erro ao iniciar câmera: {str(e)}")
            # messagebox.showerror("Erro", f"Erro ao iniciar câmera para inspeção manual: {str(e)}")
    
    def stop_live_capture_manual_inspection(self):
        """Para a captura contínua da câmera para inspeção manual."""
        try:
            # Verifica se os atributos existem antes de acessá-los
            if hasattr(self, 'live_capture'):
                self.live_capture = False
            
            if hasattr(self, 'manual_inspection_mode'):
                self.manual_inspection_mode = False
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Libera a câmera se existir e não estiver sendo usada pelo live_view
            if hasattr(self, 'camera') and self.camera is not None and not self.live_view:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera: {release_error}")
            
            # Limpa o frame mais recente
            if hasattr(self, 'latest_frame'):
                self.latest_frame = None
            
            # Remove o bind da tecla Enter
            if hasattr(self, 'master'):
                try:
                    self.master.unbind('<Return>')
                except Exception as unbind_error:
                    print(f"Erro ao remover bind da tecla Enter: {unbind_error}")
            
            # Referência ao botão removido - btn_continuous_inspect
            
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set("Câmera desconectada")
                
        except Exception as e:
            print(f"Erro ao parar captura para inspeção manual: {e}")
            # Não exibe messagebox para evitar interrupção da interface
    
    def toggle_live_capture_manual_inspection(self):
        """Alterna entre iniciar e parar a captura contínua para inspeção manual com Enter."""
        try:
            if not hasattr(self, 'live_capture'):
                self.live_capture = False
                
            if not self.live_capture:
                # Verifica se há um modelo carregado para inspeção
                if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                    self.status_var.set("É necessário carregar um modelo de inspeção antes de iniciar a captura")
                    return
                    
                self.start_live_capture_manual_inspection()
                # O texto do botão e status são atualizados na função start_live_capture_manual_inspection
            else:
                self.stop_live_capture_manual_inspection()
                # O texto do botão e status são atualizados na função stop_live_capture_manual_inspection
        except Exception as e:
            print(f"Erro ao alternar modo de inspeção manual: {e}")
            self.status_var.set(f"Erro ao alternar modo de inspeção: {str(e)}")
            # Referência ao botão removido - btn_continuous_inspect
            if hasattr(self, 'status_var'):
                self.status_var.set("Erro ao iniciar captura")
    
    def process_live_frame_manual_inspection(self):
        """Processa frames da câmera em segundo plano para inspeção manual (apenas captura, sem exibição ao vivo)."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        if not hasattr(self, 'camera') or not self.camera:
            return
            
        if not hasattr(self, 'manual_inspection_mode') or not self.manual_inspection_mode:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
                
                # NÃO atualiza a exibição automaticamente - apenas mantém o frame mais recente
                # A exibição será atualizada apenas quando Enter for pressionado
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            try:
                self.stop_live_capture_manual_inspection()
            except Exception as stop_error:
                print(f"Erro ao parar captura após falha: {stop_error}")
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if hasattr(self, 'live_capture') and self.live_capture and hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            self.master.after(100, self.process_live_frame_manual_inspection)
    
    def on_enter_key_inspection(self, event=None):
        """Manipulador de evento para a tecla Enter durante a inspeção manual."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'manual_inspection_mode') or not self.manual_inspection_mode:
            return
            
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
        
        try:
            if not hasattr(self, 'latest_frame') or self.latest_frame is None:
                self.status_var.set("Nenhum frame disponível para inspeção")
                return
                
            # Usa o frame mais recente para inspeção
            self.img_test = self.latest_frame.copy()
            
            # Salva a imagem no histórico de fotos
            try:
                self.save_to_photo_history(self.img_test)
            except Exception as save_error:
                print(f"Erro ao salvar no histórico: {save_error}")
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            try:
                self.run_inspection()
            except Exception as inspect_error:
                print(f"Erro durante inspeção: {inspect_error}")
                self.status_var.set(f"Erro durante inspeção: {str(inspect_error)}")
            
            # Atualiza status
            self.status_var.set("Inspeção realizada - Pressione ENTER para nova inspeção")
        except Exception as e:
            print(f"Erro ao realizar inspeção manual: {e}")
            self.status_var.set(f"Erro ao realizar inspeção manual: {str(e)}")
            
    def on_enter_key_continuous_inspection(self, event=None):
        """Manipulador de evento para a tecla Enter durante a inspeção contínua."""
        # Verifica se está no modo de inspeção contínua
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        # Verifica se não está no modo de inspeção manual
        if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            return
        
        try:
            if not hasattr(self, 'latest_frame') or self.latest_frame is None:
                self.status_var.set("Nenhum frame disponível para inspeção")
                return
                
            # Usa o frame mais recente para inspeção
            self.img_test = self.latest_frame.copy()
            
            # Salva a imagem no histórico de fotos
            try:
                self.save_to_photo_history(self.img_test)
            except Exception as save_error:
                print(f"Erro ao salvar no histórico: {save_error}")
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            try:
                self.run_inspection()
            except Exception as inspect_error:
                print(f"Erro durante inspeção: {inspect_error}")
                self.status_var.set(f"Erro durante inspeção: {str(inspect_error)}")
            
            # Atualiza status
            self.status_var.set("Inspeção realizada - Pressione ENTER para nova inspeção")
        except Exception as e:
            print(f"Erro ao realizar inspeção contínua: {e}")
            self.status_var.set(f"Erro ao realizar inspeção contínua: {str(e)}")
    
    def inspect_without_capture(self):
        """Executa inspeção na imagem atual sem capturar uma nova foto."""
        try:
            # Verifica se há uma imagem carregada
            if not hasattr(self, 'img_test') or self.img_test is None:
                self.status_var.set("Nenhuma imagem disponível para inspeção")
                return
                
            # Verifica se há um modelo carregado
            if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                self.status_var.set("É necessário carregar um modelo de inspeção")
                return
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            self.run_inspection()
            
            # Atualiza status
            self.status_var.set("Inspeção realizada com sucesso")
            
        except Exception as e:
            print(f"Erro ao inspecionar sem capturar: {e}")
            self.status_var.set(f"Erro ao inspecionar: {str(e)}")
    
    def show_fullscreen_image(self):
        """Exibe a imagem atual em tela cheia temporariamente."""
        try:
            if not hasattr(self, 'img_test') or self.img_test is None:
                return
                
            # Obtém dimensões do canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Usa a função cv2_to_tk para manter consistência com o resto do código
            self.img_display, self.scale_factor = cv2_to_tk(self.img_test, max_w=canvas_width, max_h=canvas_height)
            
            if self.img_display is None:
                return
            
            # Calcula dimensões da imagem redimensionada
            img_height, img_width = self.img_test.shape[:2]
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            # Limpa o canvas e exibe a imagem centralizada
            self.canvas.delete("all")
            self.x_offset = max(0, (self.canvas.winfo_width() - new_width) // 2)
            self.y_offset = max(0, (self.canvas.winfo_height() - new_height) // 2)
            self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
            
            # Atualiza o canvas
            self.canvas.update()
            
            # Aguarda um momento para que o usuário veja a imagem
            self.master.update()
            
        except Exception as e:
            print(f"Erro ao exibir imagem em tela cheia: {e}")
    
    def toggle_live_capture_inspection(self):
        """Alterna entre iniciar e parar a captura contínua para inspeção automática."""
        # Redireciona para a função de inspeção manual com Enter, que agora é a única forma de inspeção
        self.toggle_live_capture_manual_inspection()
    
    def process_live_frame_inspection(self):
        """Processa frames da câmera em segundo plano para inspeção (apenas captura, sem exibição ao vivo)."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        if not hasattr(self, 'camera') or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
                
                # NÃO atualiza a exibição automaticamente - apenas mantém o frame mais recente
                # A exibição e inspeção serão executadas apenas quando Enter for pressionado
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            try:
                self.stop_live_capture_inspection()
            except Exception as stop_error:
                print(f"Erro ao parar captura após falha: {stop_error}")
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if hasattr(self, 'live_capture') and self.live_capture:
            self.master.after(100, self.process_live_frame_inspection)
    
    def capture_test_from_webcam(self):
        """Captura instantânea da imagem mais recente da câmera para inspeção."""
        try:
            if not self.live_capture or self.latest_frame is None:
                # Fallback para captura única se não há captura contínua
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                captured_image = capture_image_from_camera(camera_index)
            else:
                # Usa o frame mais recente da captura contínua
                captured_image = self.latest_frame.copy()
            
            if captured_image is not None:
                # Para de captura ao vivo se estiver ativa
                if self.live_view:
                    self.stop_live_view()
                
                # Carrega a imagem capturada
                self.img_test = captured_image
                
                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                # Atualiza estado dos botões
                self.update_button_states()
                
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                self.status_var.set(f"Imagem capturada da câmera {camera_index}")
                
                # Salva a imagem no histórico de fotos
                self.save_to_photo_history(captured_image)
                
                # Exibe a imagem em tela cheia
                self.show_fullscreen_image()
                
                # Executa inspeção automática se modelo carregado
                if hasattr(self, 'slots') and self.slots and hasattr(self, 'img_reference') and self.img_reference is not None:
                    self.run_inspection()
            else:
                self.status_var.set("Nenhuma imagem disponível para captura")
                
        except Exception as e:
            print(f"Erro ao capturar da webcam: {e}")
            self.status_var.set(f"Erro ao capturar da webcam: {str(e)}")
    
    def save_to_photo_history(self, image):
        """Salva a imagem capturada no histórico de fotos."""
        try:
            # Cria o diretório de histórico se não existir
            historico_dir = MODEL_DIR / "historico_fotos"
            historico_dir.mkdir(exist_ok=True)
            
            # Cria diretório para capturas manuais
            capturas_dir = historico_dir / "Capturas"
            capturas_dir.mkdir(exist_ok=True)
            
            # Gera nome de arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Obtém o modelo atual se disponível para incluir no nome do arquivo
            model_name = "sem_modelo"
            model_id = getattr(self, 'current_model_id', '--')
            if model_id != '--' and hasattr(self, 'db_manager'):
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info and 'nome' in model_info:
                        model_name = model_info['nome']
                        # Substitui caracteres inválidos para nome de arquivo
                        model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                except Exception as e:
                    print(f"Erro ao obter informações do modelo: {e}")
            
            file_name = f"foto_{model_name}_{timestamp}.png"
            file_path = capturas_dir / file_name
            
            # Salva a imagem
            cv2.imwrite(str(file_path), image)
            
            print(f"Foto salva no histórico: {file_path}")
        except Exception as e:
            print(f"Erro ao salvar foto no histórico: {e}")
    
    def save_inspection_result_to_history(self, status, passed, total):
        """Salva a imagem com os resultados da inspeção no histórico de fotos."""
        try:
            if self.img_test is None:
                return
                
            # Cria o diretório de histórico se não existir
            historico_dir = MODEL_DIR / "historico_fotos"
            historico_dir.mkdir(exist_ok=True)
            
            # Cria diretórios separados para OK e NG
            ok_dir = historico_dir / "OK"
            ng_dir = historico_dir / "NG"
            ok_dir.mkdir(exist_ok=True)
            ng_dir.mkdir(exist_ok=True)
            
            # Cria uma cópia da imagem para adicionar anotações
            img_result = self.img_test.copy()
            
            # Adiciona informações da inspeção na imagem
            # Obtém o modelo atual se disponível
            model_id = getattr(self, 'current_model_id', '--')
            model_name = "--"
            if hasattr(self, 'db_manager') and model_id != '--':
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info:
                        model_name = model_info['nome']
                except:
                    pass
            
            # Adiciona texto com informações da inspeção
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(img_result, f"Data: {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_result, f"Modelo: {model_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Cor baseada no resultado
            result_color = (0, 255, 0) if status == "APROVADO" else (0, 0, 255)
            cv2.putText(img_result, f"Resultado: {status}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
            cv2.putText(img_result, f"Slots OK: {passed}/{total}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Desenha os resultados dos slots na imagem
            for result in self.inspection_results:
                is_ok = result['passou']
                corners = result['corners']
                bbox = result['bbox']
                slot_id = result['slot_id']
                
                # Cores baseadas no resultado
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                
                if corners is not None:
                    # Desenha polígono transformado
                    corners_array = np.array(corners, dtype=np.int32)
                    cv2.polylines(img_result, [corners_array], True, color, 2)
                    
                    # Adiciona texto com ID do slot e resultado
                    x, y = corners[0]
                    status_text = "OK" if is_ok else "NG"
                    cv2.putText(img_result, f"S{slot_id}: {status_text}", 
                               (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)
                elif bbox != [0,0,0,0]:  # Fallback para bbox
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
                    # Texto de erro removido - apenas retângulo é exibido
            
            # Gera nome de arquivo com timestamp e resultado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_tag = "OK" if status == "APROVADO" else "NG"
            file_name = f"inspecao_{timestamp}.png"
            
            # Seleciona o diretório correto com base no resultado
            target_dir = ok_dir if status == "APROVADO" else ng_dir
            
            # Obtém o modelo atual se disponível para incluir no nome do arquivo
            model_id = getattr(self, 'current_model_id', '--')
            if model_id != '--' and hasattr(self, 'db_manager'):
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info and 'nome' in model_info:
                        model_name = model_info['nome']
                        # Substitui caracteres inválidos para nome de arquivo
                        model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        file_name = f"inspecao_{model_name}_{timestamp}.png"
                except Exception as e:
                    print(f"Erro ao obter informações do modelo: {e}")
            
            file_path = target_dir / file_name
            
            # Salva a imagem
            cv2.imwrite(str(file_path), img_result)
            
            print(f"Resultado de inspeção salvo no histórico: {file_path}")
        except Exception as e:
            print(f"Erro ao salvar resultado de inspeção no histórico: {e}")
    

    
    def stop_live_view(self):
        """Para a captura ao vivo."""
        try:
            # Verifica se o atributo live_view existe
            if hasattr(self, 'live_view'):
                self.live_view = False
            
            # Libera a câmera se existir
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera no stop_live_view: {release_error}")
                    
        except Exception as e:
            print(f"Erro ao parar visualização ao vivo: {e}")
            # Não exibe messagebox para evitar interrupção da interface
    
    def process_live_frame(self):
        """Processa frame da câmera de forma otimizada"""
        try:
            # Verifica se os atributos necessários existem
            if not hasattr(self, 'live_view') or not hasattr(self, 'camera'):
                return
                
            if not self.live_view or not self.camera:
                return
            
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Atualiza a imagem de teste
                    if hasattr(self, 'img_test'):
                        self.img_test = frame
                    
                    # Atualiza o display
                    try:
                        self.update_display()
                    except Exception as display_error:
                        print(f"Erro ao atualizar display: {display_error}")
                    
                    # Inspeção automática otimizada (menos frequente)
                    if hasattr(self, 'slots') and self.slots and hasattr(self, '_frame_count'):
                        self._frame_count += 1
                        # Executa inspeção a cada 5 frames para melhor performance
                        if self._frame_count % 5 == 0:
                            try:
                                self.run_inspection(show_message=False)
                            except Exception as inspection_error:
                                print(f"Erro durante inspeção automática: {inspection_error}")
                    elif hasattr(self, 'slots') and self.slots:
                        self._frame_count = 0
            except Exception as camera_error:
                print(f"Erro ao ler frame da câmera: {camera_error}")
            
            # Agenda próximo frame
            if hasattr(self, 'live_view') and self.live_view and hasattr(self, 'master'):
                try:
                    self.master.after(100, self.process_live_frame)
                except Exception as schedule_error:
                    print(f"Erro ao agendar próximo frame: {schedule_error}")
                    
        except Exception as e:
            print(f"Erro geral no processamento de frame: {e}")
            # Tenta agendar o próximo frame mesmo com erro para manter a continuidade
            if hasattr(self, 'master') and hasattr(self, 'live_view') and self.live_view:
                try:
                    self.master.after(100, self.process_live_frame)
                except Exception:
                    pass  # Ignora erro no agendamento de recuperação
    
    def update_display(self):
        """Atualiza exibição no canvas de forma otimizada"""
        try:
            # Verifica se os atributos necessários existem
            if not hasattr(self, 'img_test') or self.img_test is None:
                return
                
            if not hasattr(self, 'canvas'):
                print("Erro: Canvas não encontrado")
                return
            
            # === AJUSTE AUTOMÁTICO AO CANVAS ===
            try:
                # Obtém o tamanho atual do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # Se o canvas ainda não foi renderizado, use valores padrão
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 640
                    canvas_height = 480
            except Exception as canvas_error:
                print(f"Erro ao obter dimensões do canvas: {canvas_error}")
                canvas_width = 640
                canvas_height = 480
            
            # Converte a imagem para o tamanho do canvas
            try:
                self.img_display, self.scale_factor = cv2_to_tk(self.img_test, max_w=canvas_width, max_h=canvas_height)
            except Exception as convert_error:
                print(f"Erro ao converter imagem para exibição: {convert_error}")
                return
            
            if self.img_display is None:
                return
            
            # === ATUALIZAÇÃO EFICIENTE DO CANVAS ===
            try:
                # Remove apenas overlays, mantém imagem base quando possível
                self.canvas.delete("result_overlay")
                self.canvas.delete("inspection")
                
                # Calcula dimensões da imagem redimensionada e offsets para centralização
                img_height, img_width = self.img_test.shape[:2]
                new_width = int(img_width * self.scale_factor)
                new_height = int(img_height * self.scale_factor)
                self.x_offset = max(0, (self.canvas.winfo_width() - new_width) // 2)
                self.y_offset = max(0, (self.canvas.winfo_height() - new_height) // 2)
                
                # Cria ou atualiza imagem
                if not hasattr(self, '_canvas_image_id'):
                    self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
                else:
                    self.canvas.itemconfig(self._canvas_image_id, image=self.img_display)
                    self.canvas.coords(self._canvas_image_id, self.x_offset, self.y_offset)
            except Exception as canvas_update_error:
                print(f"Erro ao atualizar canvas: {canvas_update_error}")
                return
            
            # Desenha resultados se disponíveis
            if hasattr(self, 'inspection_results') and self.inspection_results:
                try:
                    self.draw_inspection_results()
                except Exception as draw_error:
                    print(f"Erro ao desenhar resultados de inspeção: {draw_error}")
            
            # Atualiza scroll region apenas se necessário
            try:
                bbox = self.canvas.bbox("all")
            except Exception as bbox_error:
                print(f"Erro ao obter bbox do canvas: {bbox_error}")
                return
                if bbox != self.canvas.cget("scrollregion"):
                    try:
                        self.canvas.configure(scrollregion=bbox)
                    except Exception as scroll_error:
                        print(f"Erro ao configurar região de scroll: {scroll_error}")
        except Exception as e:
            print(f"Erro geral ao atualizar display: {e}")
    
    def run_inspection(self, show_message=False):
        """Executa inspeção otimizada com estilo industrial Keyence"""
        try:
            # === ATUALIZAÇÃO DE STATUS ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("PROCESSANDO...")
                    if hasattr(self, 'update_idletasks'):
                        self.update_idletasks()  # Força atualização da UI
            except Exception as e:
                print(f"Erro ao atualizar status: {e}")
            
            # === VALIDAÇÃO INICIAL ===
            if not hasattr(self, 'slots') or not self.slots or \
               not hasattr(self, 'img_reference') or self.img_reference is None or \
               not hasattr(self, 'img_test') or self.img_test is None:
                if hasattr(self, 'status_var'):
                    self.status_var.set("Carregue o modelo de referência E a imagem de teste antes de inspecionar")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                return
            
            print("--- Iniciando Inspeção Keyence ---")
            
            # Limpa resultados anteriores
            if hasattr(self, 'canvas'):
                try:
                    self.canvas.delete("result_overlay")
                except Exception as canvas_error:
                    print(f"Erro ao limpar canvas: {canvas_error}")
            
            # === 1. ALINHAMENTO DE IMAGEM ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ALINHANDO...")
                if hasattr(self, 'update_idletasks'):
                    self.update_idletasks()  # Força atualização da UI
                M, _, align_error = find_image_transform(self.img_reference, self.img_test)
            except Exception as e:
                print(f"Erro durante alinhamento: {e}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Erro durante alinhamento: {e}")
                return
            
            if M is None:
                print(f"FALHA no Alinhamento: {align_error}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("FALHA DE ALINHAMENTO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Falha no Alinhamento: Não foi possível alinhar as imagens. Erro: {align_error}")
                
                # Desenha slots de referência em cor de erro (estilo Keyence)
                if hasattr(self, 'canvas') and hasattr(self, 'scale_factor'):
                    try:
                        for slot in self.slots:
                            xr, yr, wr, hr = slot['x'], slot['y'], slot['w'], slot['h']
                            xa, ya = xr * self.scale_factor + self.x_offset, yr * self.scale_factor + self.y_offset
                            wa, ha = wr * self.scale_factor, hr * self.scale_factor
                            self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=COLOR_ALIGN_FAIL, width=2, tags="result_overlay")
                            # Carrega as configurações de estilo
                            try:
                                style_config = load_style_config()
                                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot['id']}\nFAIL", fill=COLOR_ALIGN_FAIL, font=style_config["ng_font"], tags="result_overlay", justify="center")
                            except Exception as style_error:
                                print(f"Erro ao carregar configurações de estilo: {style_error}")
                                # Fallback para fonte padrão
                                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot['id']}\nFAIL", fill=COLOR_ALIGN_FAIL, tags="result_overlay", justify="center")
                    except Exception as draw_error:
                        print(f"Erro ao desenhar slots de referência: {draw_error}")
                return
            
            # === 2. VERIFICAÇÃO DOS SLOTS (ESTILO KEYENCE) ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("INSPECIONANDO...")
                if hasattr(self, 'update_idletasks'):
                    self.update_idletasks()  # Força atualização da UI
                
                overall_ok = True
                self.inspection_results = []
                failed_slots = []  # Para log otimizado
                
                # Adicionar modelo_id aos resultados se disponível
                model_id = getattr(self, 'current_model_id', '--')
                
                for i, slot in enumerate(self.slots):
                    # Atualizar status com progresso
                    progress = f"SLOT {i+1}/{len(self.slots)}"
                    if hasattr(self, 'inspection_status_var'):
                        self.inspection_status_var.set(progress)
                    if hasattr(self, 'update_idletasks'):
                        self.update_idletasks()  # Força atualização da UI
                    
                    try:
                        # Processamento otimizado sem logs excessivos
                        is_ok, correlation, pixels, corners, bbox, log_msgs = check_slot(self.img_test, slot, M)
                        
                        # Log apenas para falhas (reduz overhead)
                        if not is_ok:
                            failed_slots.append(f"S{slot['id']}({slot['tipo']})")
                            for msg in log_msgs:
                                print(f"  -> {msg}")
                        
                        # Armazena resultado otimizado com estilo Keyence
                        result = {
                            'slot_id': slot['id'],
                            'passou': is_ok,
                            'score': correlation,
                            'detalhes': f"Score: {correlation:.3f}, Pixels: {pixels}",
                            'slot_data': slot,
                            'corners': corners,
                            'bbox': bbox,
                            'model_id': model_id
                        }
                        self.inspection_results.append(result)
                        
                        if not is_ok:
                            overall_ok = False
                    except Exception as slot_error:
                        print(f"Erro ao processar slot {slot['id']}: {slot_error}")
                        # Continua com o próximo slot em caso de erro
            except Exception as e:
                print(f"Erro durante inspeção: {e}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Erro durante inspeção: {e}")
                return
            
            # === 3. DESENHO OTIMIZADO NO CANVAS COM ESTILO KEYENCE ===
            try:
                if not hasattr(self, 'canvas') or not hasattr(self, 'inspection_results') or not hasattr(self, 'scale_factor'):
                    print("Atributos necessários para desenho não estão disponíveis")
                    return
                    
                for result in self.inspection_results:
                    try:
                        is_ok = result.get('passou', False)
                        corners = result.get('corners', None)
                        bbox = result.get('bbox', [0,0,0,0])
                        slot_id = result.get('slot_id', '?')
                        
                        # Cores no estilo Keyence
                        fill_color = COLOR_PASS if is_ok else COLOR_FAIL
                        
                        if corners is not None:
                            try:
                                # Conversão otimizada de coordenadas
                                canvas_corners = [(int(pt[0] * self.scale_factor) + self.x_offset, int(pt[1] * self.scale_factor) + self.y_offset) for pt in corners]
                                
                                # Desenha polígono transformado estilo Keyence
                                self.canvas.create_polygon(canvas_corners, outline=fill_color, fill="", width=2, tags="result_overlay")
                                
                                # Adiciona um pequeno retângulo de status no canto estilo Keyence
                                status_x, status_y = canvas_corners[0][0], canvas_corners[0][1] - 20
                                self.canvas.create_rectangle(status_x, status_y, status_x + 40, status_y + 16, 
                                                           fill=fill_color, outline="", tags="result_overlay")
                                
                                # Label otimizado estilo Keyence
                                try:
                                    # Carrega as configurações de estilo
                                    style_config = load_style_config()
                                    self.canvas.create_text(status_x + 20, status_y + 8,
                                                          text=f"S{slot_id}", fill="#FFFFFF", anchor="center", tags="result_overlay",
                                                          font=style_config["ok_font"])
                                    
                                    # Adiciona indicador de status
                                    status_text = "OK" if is_ok else "NG"
                                    # Escolhe a fonte baseada no resultado
                                    font_str = style_config["ok_font"] if is_ok else style_config["ng_font"]
                                    self.canvas.create_text(canvas_corners[0][0] + 60, canvas_corners[0][1] - 12,
                                                          text=status_text, fill=fill_color, anchor="nw", tags="result_overlay",
                                                          font=font_str)
                                except Exception as style_error:
                                    print(f"Erro ao carregar estilo ou criar texto: {style_error}")
                                    # Fallback para texto simples sem estilo
                                    self.canvas.create_text(status_x + 20, status_y + 8,
                                                          text=f"S{slot_id}", fill="#FFFFFF", anchor="center", tags="result_overlay")
                                    self.canvas.create_text(canvas_corners[0][0] + 60, canvas_corners[0][1] - 12,
                                                          text="OK" if is_ok else "NG", fill=fill_color, anchor="nw", tags="result_overlay")
                            except Exception as corner_error:
                                print(f"Erro ao processar corners para slot {slot_id}: {corner_error}")
                        
                        if bbox != [0,0,0,0]:  # Fallback para bbox com estilo Keyence
                                try:
                                    xa, ya = bbox[0] * self.scale_factor, bbox[1] * self.scale_factor
                                    wa, ha = bbox[2] * self.scale_factor, bbox[3] * self.scale_factor
                                    # Linha pontilhada de erro removida conforme solicitado pelo usuário
                                    # Indicador de erro estilo Keyence removido conforme solicitado pelo usuário.
                                except Exception as bbox_error:
                                    print(f"Erro ao processar bbox para slot {slot_id}: {bbox_error}")
                    except Exception as result_error:
                        print(f"Erro ao processar resultado de inspeção: {result_error}")
                        continue  # Continua com o próximo resultado
            except Exception as draw_error:
                print(f"Erro ao desenhar resultados no canvas: {draw_error}")
            # Continua com o processamento para atualizar o status
            
            # === 4. RESULTADO FINAL ESTILO KEYENCE ===
            try:
                if not hasattr(self, 'inspection_results'):
                    print("Resultados de inspeção não disponíveis")
                    return
                    
                total = len(self.inspection_results)
                passed = sum(1 for r in self.inspection_results if r.get('passou', False))
                
                failed = total - passed
            except Exception as count_error:
                print(f"Erro ao contar resultados de inspeção: {count_error}")
                return
            final_status = "APROVADO" if overall_ok else "REPROVADO"
            
            # Atualizar status de inspeção
            if hasattr(self, 'inspection_status_var'):
                try:
                    if overall_ok:
                        self.inspection_status_var.set("OK")
                    else:
                        self.inspection_status_var.set("NG")
                except Exception as status_error:
                    print(f"Erro ao atualizar status de inspeção: {status_error}")
            
            # Log otimizado estilo Keyence
            if failed_slots:
                print(f"Falhas detectadas em: {', '.join(failed_slots)}")
            print(f"--- Inspeção Keyence Concluída: {final_status} ({passed}/{total}) ---")
            
            # Atualiza interface com estilo industrial Keyence
            if hasattr(self, 'update_results_list') and callable(self.update_results_list):
                try:
                    self.update_results_list()
                except Exception as update_error:
                    print(f"Erro ao atualizar lista de resultados: {update_error}")
            
            # Salva a imagem com os resultados da inspeção no histórico
            if hasattr(self, 'save_inspection_result_to_history') and callable(self.save_inspection_result_to_history):
                try:
                    self.save_inspection_result_to_history(final_status, passed, total)
                except Exception as save_error:
                    print(f"Erro ao salvar resultado no histórico: {save_error}")
            
            # Status com estilo industrial Keyence
            status_text = f"INSPEÇÃO: {final_status} - {passed}/{total} SLOTS OK, {failed} FALHAS"
            if hasattr(self, 'status_var'):
                try:
                    self.status_var.set(status_text)
                except Exception as status_var_error:
                    print(f"Erro ao atualizar texto de status: {status_var_error}")
            
            # Atualiza cor da barra de status baseado no resultado estilo Keyence
            try:
                # Armazenamos uma referência direta ao status_bar durante a criação
                if hasattr(self, 'status_bar'):
                    if overall_ok:
                        self.status_bar.config(background="#00AA00", foreground="#FFFFFF")
                    else:
                        self.status_bar.config(background="#CC0000", foreground="#FFFFFF")
                        
                # Atualizar cor do indicador de status de inspeção usando referência direta
                if hasattr(self, 'inspection_status_label'):
                    if overall_ok:
                        self.inspection_status_label.config(foreground="#00AA00")
                    else:
                        self.inspection_status_label.config(foreground="#CC0000")
            except Exception as e:
                print(f"Erro ao atualizar status_bar: {e}")
            
            # Não exibimos mais mensagens, apenas atualizamos o status
            # O status já foi atualizado acima com o texto: f"INSPEÇÃO: {final_status} - {passed}/{total} SLOTS OK, {failed} FALHAS"
        except Exception as final_error:
            print(f"Erro ao processar resultado final: {final_error}")
    
    def create_status_summary_panel(self, parent_frame=None):
        """Cria o painel de resumo de status estilo Keyence IV3"""
        # Se um frame pai for fornecido, criar um painel de resumo geral
        if parent_frame:
            # Frame para o painel de status geral
            status_panel = ttk.Frame(parent_frame, relief="raised", borderwidth=2)
            status_panel.pack(fill=X, pady=5)
            
            # Linha 1: Status geral
            status_row = ttk.Frame(status_panel)
            status_row.pack(fill=X, pady=2)
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            ttk.Label(status_row, text="STATUS:", font=style_config["ok_font"]).pack(side=LEFT, padx=(5, 5))
            
            # Label para status (OK/NG) com estilo industrial Keyence
            self.status_label = ttk.Label(status_row, text="--", font=style_config["ok_font"], 
                                        background="#333333", foreground="#FFFFFF", 
                                        width=6, anchor="center", padding=3)
            self.status_label.pack(side=LEFT, padx=5)
            
            # Linha 2: Score e ID
            details_row = ttk.Frame(status_panel)
            details_row.pack(fill=X, pady=2)
            
            # Usa as configurações de estilo já carregadas
            ttk.Label(details_row, text="SCORE:", font=style_config["ok_font"]).pack(side=LEFT, padx=(5, 5))
            
            # Label para score com estilo industrial Keyence
            self.score_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                       background="#333333", foreground="#FFFFFF", 
                                       width=8, anchor="center", padding=3)
            self.score_label.pack(side=LEFT, padx=5)
            
            ttk.Label(details_row, text="ID:", font=style_config["ok_font"]).pack(side=LEFT, padx=(10, 5))
            
            # Label para ID do modelo com estilo industrial Keyence
            self.id_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                    background="#333333", foreground="#FFFFFF", 
                                    anchor="center", padding=3)
            self.id_label.pack(side=LEFT, padx=5, fill=X, expand=True)
            return
        
        # Caso contrário, estamos criando o painel principal de status
        # Primeiro, limpe qualquer widget existente no status_grid_frame
        for widget in self.status_grid_frame.winfo_children():
            widget.destroy()
            
        # Vamos adicionar um cabeçalho mais proeminente
        header_frame = ttk.Frame(self.status_grid_frame)
        header_frame.pack(fill=X, pady=(0, 10))
        
        # Título do painel de status
        header_label = ttk.Label(header_frame, text="SISTEMA DE INSPEÇÃO WALTER", 
                                font=('Arial', 14, 'bold'), foreground="white",
                                background=self.accent_color)
        header_label.pack(pady=10, fill=X)
        
        # Caso contrário, criar o painel de resumo de slots
        # Limpar widgets existentes
        for widget in self.status_widgets.values():
            if hasattr(widget, 'frame'):
                widget['frame'].destroy()
        self.status_widgets.clear()
        
        if not self.slots:
            return
        
        # Criar um frame para conter os slots usando pack em vez de grid
        slots_container = ttk.Frame(self.status_grid_frame)
        slots_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Calcular layout (máximo 6 colunas)
        num_slots = len(self.slots)
        cols = min(6, num_slots)
        
        # Criar frames para cada coluna
        column_frames = []
        for i in range(cols):
            col_frame = ttk.Frame(slots_container)
            col_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=2)
            column_frames.append(col_frame)
        
        # Distribuir slots pelas colunas
        for i, slot in enumerate(self.slots):
            col_idx = i % cols
            
            # Frame para cada slot com estilo industrial
            slot_frame = ttk.Frame(column_frames[col_idx], relief="raised", borderwidth=2)
            slot_frame.pack(fill=X, pady=3, padx=2)
            
            # Label do ID do slot com estilo industrial Keyence
            id_label = ttk.Label(slot_frame, text=f"SLOT {slot['id']}", 
                                font=('Arial', 8, 'bold'), background="#1E1E1E", foreground="#FFFFFF")
            id_label.pack(pady=2, fill=X)
            
            # Label do status (OK/NG) com estilo industrial Keyence
            status_label = ttk.Label(slot_frame, text="---", 
                                   font=('Arial', 14, 'bold'),
                                   foreground="#7F8C8D",
                                   background="#2A2A2A",
                                   anchor="center")
            status_label.pack(pady=2, fill=X)
            
            # Label do score com estilo industrial Keyence
            score_label = ttk.Label(slot_frame, text="", 
                                  font=('Arial', 9, 'bold'),
                                  background="#1E1E1E",
                                  foreground="#CCCCCC")
            score_label.pack(pady=1, fill=X)
            
            # Armazenar referências
            self.status_widgets[slot['id']] = {
                'frame': slot_frame,
                'id_label': id_label,
                'status_label': status_label,
                'score_label': score_label
            }
    
    def update_status_summary_panel(self):
        """Atualiza o painel de resumo com os resultados da inspeção no estilo industrial"""
        if not hasattr(self, 'status_widgets') or not self.status_widgets:
            return
        
        # Resetar todos os status com estilo industrial
        for slot_id, widgets in self.status_widgets.items():
            if all(key in widgets for key in ['status_label', 'score_label', 'frame']):
                try:
                    widgets['status_label'].config(text="---", foreground="#7F8C8D", background="#2A2A2A")
                    widgets['score_label'].config(text="---", background="#1E1E1E", foreground="#CCCCCC")
                    widgets['frame'].config(relief="raised", borderwidth=2, padding=2)
                except Exception as e:
                    print(f"Erro ao resetar widget do slot {slot_id}: {e}")
        
        # Verificar se temos resultados de inspeção
        if not hasattr(self, 'inspection_results') or not self.inspection_results:
            return
            
        # Atualizar com resultados da inspeção usando estilo industrial
        for result in self.inspection_results:
            slot_id = result['slot_id']
            if slot_id in self.status_widgets:
                widgets = self.status_widgets[slot_id]
                
                # Carrega as configurações de estilo
                style_config = load_style_config()
                
                try:
                    if result['passou']:
                        # Estilo industrial para OK (cor personalizada)
                        widgets['status_label'].config(text="OK", foreground="#FFFFFF", background=style_config["ok_color"])
                        widgets['frame'].config(relief="raised", borderwidth=3)
                        widgets['id_label'].config(background="#003300", foreground="#FFFFFF")
                    else:
                        # Estilo industrial para NG (cor personalizada)
                        widgets['status_label'].config(text="NG", foreground="#FFFFFF", background=style_config["ng_color"])
                        widgets['frame'].config(relief="raised", borderwidth=3)
                        widgets['id_label'].config(background="#330000", foreground="#FFFFFF")
                    
                    # Atualizar score com estilo industrial
                    score_text = f"{result['score']:.3f}"
                    if result['passou']:
                        widgets['score_label'].config(text=score_text, background="#003300", foreground="#FFFFFF")
                    else:
                        widgets['score_label'].config(text=score_text, background="#330000", foreground="#FFFFFF")
                except Exception as e:
                    print(f"Erro ao atualizar widget do slot {slot_id}: {e}")
    
    def update_results_list(self):
        """Atualiza lista de resultados com estilo industrial Keyence"""
        # === LIMPEZA OTIMIZADA ===
        children = self.results_listbox.get_children()
        if children:
            self.results_listbox.delete(*children)  # Mais eficiente que loop
        
        # === CONFIGURAÇÃO DE TAGS ESTILO KEYENCE ===
        # Carrega as configurações de estilo
        style_config = load_style_config()
        
        # Estilo OK - cor personalizada
        self.results_listbox.tag_configure("pass", 
                                         foreground="#FFFFFF", 
                                         background=style_config["ok_color"], 
                                         font=style_config["ok_font"])
        
        # Estilo NG - cor personalizada
        self.results_listbox.tag_configure("fail", 
                                         foreground="#FFFFFF", 
                                         background=style_config["ng_color"], 
                                         font=style_config["ng_font"])
        
        # Estilo cabeçalho - cinza industrial Keyence
        # Carrega as configurações de estilo
        style_config = load_style_config()
        self.results_listbox.tag_configure("header", 
                                         foreground="#FFFFFF", 
                                         background="#333333", 
                                         font=style_config["ok_font"])
        
        # === VARIÁVEIS PARA RESUMO GERAL ===
        total_slots = len(self.inspection_results) if self.inspection_results else 0
        passed_slots = 0
        total_score = 0
        model_id = "--"
        
        # === INSERÇÃO OTIMIZADA COM ESTILO INDUSTRIAL KEYENCE ===
        for result in self.inspection_results:
            status = "OK" if result['passou'] else "NG"
            score_text = f"{result['score']:.3f}"
            tags = ("pass",) if result['passou'] else ("fail",)
            
            # Atualizar contadores para resumo
            if result['passou']:
                passed_slots += 1
            total_score += result['score']
            
            # Obter ID do modelo se disponível
            if 'model_id' in result and model_id == "--":
                model_id = result['model_id']
            
            # Detalhes formatados para estilo industrial Keyence
            detalhes = result['detalhes'].upper() if result['passou'] else f"⚠ {result['detalhes'].upper()}"
            
            self.results_listbox.insert("", "end",
                                       text=result['slot_id'],
                                       values=(status, score_text, detalhes),
                                       tags=tags)
        
        # Atualizar painel de resumo de status detalhado
        self.update_status_summary_panel()
        
        # Atualizar painel de resumo geral se existir
        if hasattr(self, 'status_label') and hasattr(self, 'score_label') and hasattr(self, 'id_label'):
            # Calcular status geral no estilo Keyence
            if total_slots > 0:
                total_score / total_slots
                overall_status = "OK" if passed_slots == total_slots else "NG"
                
                # Atualizar labels com estilo Keyence
                self.status_label.config(
                    text=overall_status,
                    background="#00AA00" if overall_status == "OK" else "#CC0000",
                    foreground="#FFFFFF"
                )
                
                self.score_label.config(
                    text=f"{passed_slots}/{total_slots}",
                    background="#00AA00" if passed_slots == total_slots else "#CC0000",
                    foreground="#FFFFFF"
                )
                
                self.id_label.config(text=model_id)
    
    def draw_inspection_results(self):
        """Desenha resultados da inspeção no canvas com estilo industrial."""
        if not self.inspection_results:
            return
        
        for result in self.inspection_results:
            slot = result['slot_data']
            
            # Converte coordenadas da imagem para canvas (incluindo offsets)
            x1 = int(slot['x'] * self.scale_factor) + self.x_offset
            y1 = int(slot['y'] * self.scale_factor) + self.y_offset
            x2 = int((slot['x'] + slot['w']) * self.scale_factor) + self.x_offset
            y2 = int((slot['y'] + slot['h']) * self.scale_factor) + self.y_offset
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Cores estilo industrial
            if result['passou']:
                outline_color = style_config["ok_color"]  # Cor de OK personalizada
                fill_color = style_config["ok_color"]     # Mesma cor para o fundo
                text_color = "#FFFFFF"                    # Texto branco
            else:
                outline_color = style_config["ng_color"]  # Cor de NG personalizada
                fill_color = style_config["ng_color"]     # Mesma cor para o fundo
                text_color = "#FFFFFF"                    # Texto branco
            
            # Desenha retângulo com estilo industrial
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                       outline=outline_color, width=3, 
                                       dash=(3, 2) if not result['passou'] else None,
                                       tags="inspection")
            
            # Cria fundo para o texto (estilo industrial)
            text_bg_width = 60
            text_bg_height = 20
            self.canvas.create_rectangle(x1, y1, x1 + text_bg_width, y1 + text_bg_height,
                                       fill=fill_color, outline=outline_color, width=1,
                                       tags="inspection")
            
            # Adiciona texto com resultado estilo industrial
            status_text = "OK" if result['passou'] else "NG"
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Escolhe a fonte baseada no resultado
            font_str = style_config["ok_font"] if result['passou'] else style_config["ng_font"]
            
            self.canvas.create_text(x1 + text_bg_width/2, y1 + text_bg_height/2,
                                  text=f"S{slot['id']}: {status_text}",
                                  fill=text_color, font=font_str,
                                  anchor="center", tags="inspection")
            
            # Adiciona score em outra posição
            score_text = f"{result['score']:.2f}"
            # Escolhe a fonte baseada no resultado (já temos style_config carregado)
            font_str = style_config["ok_font"] if result['passou'] else style_config["ng_font"]
            self.canvas.create_text(x2 - 5, y2 - 5,
                                  text=score_text,
                                  fill=outline_color, font=font_str,
                                  anchor="se", tags="inspection")
    
    def update_button_states(self):
        """Atualiza estado dos botões baseado no estado atual."""
        len(self.slots) > 0
        self.img_test is not None
        
        # Botões que dependem de modelo e imagem de teste
        # Nota: btn_inspect foi removido, então não precisamos mais atualizar seu estado
    
    def start_background_frame_capture(self):
        """Inicia a captura contínua de frames em segundo plano."""
        def capture_frames():
            while self.live_capture and self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.latest_frame = frame.copy()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura de frame: {e}")
                    break
        
        import threading
        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()
    
    def on_closing_inspection(self):
        """Limpa recursos ao fechar a aplicação de inspeção."""
        if self.live_capture:
            self.stop_live_capture_inspection()
        if self.live_view:
            self.stop_live_view()
        self.master.destroy()


def create_main_window():
    """Cria e configura a janela principal da aplicação."""
    # Inicializa ttkbootstrap com tema mais claro
    root = ttk.Window(themename="flatly")  # Tema alterado de "cyborg" para "flatly"
    root.title("Sistema de Visão Computacional - Inspeção de Montagem")
    
    # Configurar para abrir em tela cheia
    root.state('zoomed')  # Maximiza a janela no Windows
    
    # Configura fechamento de janelas OpenCV
    def on_closing():
        cv2.destroyAllWindows()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Cria notebook para abas
    notebook = ttk.Notebook(root)
    notebook.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    # Aba Editor de Malha
    montagem_frame = MontagemWindow(notebook)
    montagem_frame.pack(fill=BOTH, expand=True)
    notebook.add(montagem_frame, text="Editor de Malha")
    
    # Aba Inspeção
    inspecao_frame = InspecaoWindow(notebook)
    inspecao_frame.pack(fill=BOTH, expand=True)
    notebook.add(inspecao_frame, text="Inspeção")
    
    # Aba Histórico de Fotos
    historico_frame = HistoricoFotosWindow(notebook)
    historico_frame.pack(fill=BOTH, expand=True)
    notebook.add(historico_frame, text="Histórico de Fotos")
    
    # Garantir que a aba "Editor de Malha" seja selecionada por padrão
    notebook.select(0)
    
    # Adicionar evento para detectar mudança de aba
    def on_tab_changed(event):
        # Verificar se a aba selecionada é a de Inspeção (índice 1)
        if notebook.index(notebook.select()) == 1:
            # Iniciar captura da câmera automaticamente
            inspecao_frame.start_live_capture_manual_inspection()
    
    # Vincular evento de mudança de aba
    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
    
    # Adicionar evento para detectar mudança de aba
    def on_tab_changed(event):
        # Verificar se a aba selecionada é a de Inspeção (índice 1)
        if notebook.index(notebook.select()) == 1:
            # Iniciar captura da câmera automaticamente
            inspecao_frame.start_live_capture_manual_inspection()
    
    # Vincular evento de mudança de aba
    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
    
    return root


class HistoricoFotosWindow(ttk.Frame):
    """Janela para exibir o histórico de fotos capturadas com data e hora."""
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Inicializar atributos para os frames scrollable e combobox
        self.todas_scrollable_frame = None
        self.ok_scrollable_frame = None
        self.ng_scrollable_frame = None
        self.capturas_scrollable_frame = None
        self.programa_combobox = None
        
        # Variáveis para filtro de programa
        self.programas_disponiveis = ["Todos"]
        self.programa_selecionado = StringVar(value="Todos")
        
        # Listas para armazenar informações das fotos por categoria
        self.fotos_ok = []
        self.fotos_ng = []
        self.fotos_capturas = []
        self.fotos_historico = []
        
        # Configuração de estilo
        self.style = ttk.Style()
        style_config = load_style_config()
        
        # Cores industriais
        self.bg_color = style_config["background_color"]
        self.panel_color = "#2A2A2A"
        self.accent_color = style_config["button_color"]
        self.text_color = style_config["text_color"]
        
        # Diretório para salvar as fotos do histórico
        self.historico_dir = MODEL_DIR / "historico_fotos"
        self.historico_dir.mkdir(exist_ok=True)
        
        # Cria diretórios separados para OK, NG e Capturas se não existirem
        self.ok_dir = self.historico_dir / "OK"
        self.ng_dir = self.historico_dir / "NG"
        self.capturas_dir = self.historico_dir / "Capturas"
        self.ok_dir.mkdir(exist_ok=True)
        self.ng_dir.mkdir(exist_ok=True)
        self.capturas_dir.mkdir(exist_ok=True)
        self.capturas_scrollable_frame = None
        
        # Configurar interface
        self.setup_ui()
        
        # Carregar fotos existentes
        self.carregar_fotos_existentes()
        
        # Configurar estilos
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TLabelframe', background=self.panel_color, borderwidth=2, relief='groove')
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent_color, 
                             font=style_config["ok_font"])
        
        # Configurar interface
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface do usuário."""
        # Limpar widgets existentes para evitar duplicação
        for widget in self.winfo_children():
            widget.destroy()
            
        # Frame principal com layout horizontal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        # Cabeçalho com título
        header_frame = ttk.Frame(left_panel, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 15))
        
        # Estilo para o cabeçalho
        self.style.configure('Header.TFrame', background=self.accent_color)
        
        # Logo e título
        header_label = ttk.Label(header_frame, text="Histórico de Fotos", 
                                font=("Arial", 12, "bold"), foreground="white",
                                background=self.accent_color)
        header_label.pack(pady=10, fill=X)
        
        # Botões de controle
        controls_frame = ttk.LabelFrame(left_panel, text="CONTROLES")
        controls_frame.pack(fill=X, pady=(0, 10))
        
        # Filtro por programa
        filter_frame = ttk.LabelFrame(controls_frame, text="FILTRAR POR PROGRAMA")
        filter_frame.pack(fill=X, padx=5, pady=5)
        
        # Combobox para seleção de programa
        self.programa_combobox = ttk.Combobox(filter_frame, 
                                           textvariable=self.programa_selecionado,
                                           state="readonly")
        self.programa_combobox.pack(fill=X, padx=5, pady=5)
        self.programa_combobox.bind("<<ComboboxSelected>>", self.filtrar_por_programa)
        
        # Botão para atualizar histórico
        self.btn_atualizar = ttk.Button(controls_frame, text="ATUALIZAR HISTÓRICO", 
                                     command=self.atualizar_historico)
        self.btn_atualizar.pack(fill=X, padx=5, pady=5)
        
        # Botão para limpar histórico
        self.btn_limpar = ttk.Button(controls_frame, text="LIMPAR HISTÓRICO", 
                                   command=self.limpar_historico)
        self.btn_limpar.pack(fill=X, padx=5, pady=5)
        
        # Painel direito - Histórico de fotos
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Notebook para organizar fotos por categoria
        self.historico_notebook = ttk.Notebook(right_panel)
        self.historico_notebook.pack(fill=BOTH, expand=True)
        
        # Aba para todas as fotos
        self.todas_fotos_frame = ttk.Frame(self.historico_notebook)
        self.historico_notebook.add(self.todas_fotos_frame, text="Todas as Fotos")
        
        # Aba para fotos OK
        self.ok_fotos_frame = ttk.Frame(self.historico_notebook)
        self.historico_notebook.add(self.ok_fotos_frame, text="Aprovadas (OK)")
        
        # Aba para fotos NG
        self.ng_fotos_frame = ttk.Frame(self.historico_notebook)
        self.historico_notebook.add(self.ng_fotos_frame, text="Reprovadas (NG)")
        
        # Aba para capturas manuais
        self.capturas_fotos_frame = ttk.Frame(self.historico_notebook)
        self.historico_notebook.add(self.capturas_fotos_frame, text="Capturas Manuais")
        
        # Criar scrollable frames para cada aba
        self.todas_scrollable_frame = self.criar_scrollable_frame(self.todas_fotos_frame, "todas")
        self.ok_scrollable_frame = self.criar_scrollable_frame(self.ok_fotos_frame, "ok")
        self.ng_scrollable_frame = self.criar_scrollable_frame(self.ng_fotos_frame, "ng")
        self.capturas_scrollable_frame = self.criar_scrollable_frame(self.capturas_fotos_frame, "capturas")
    
    def criar_scrollable_frame(self, parent_frame, categoria):
        """Cria um frame com scrollbar para exibir fotos."""
        try:
            # Limpar widgets existentes para evitar duplicação
            for widget in parent_frame.winfo_children():
                widget.destroy()
                
            canvas = Canvas(parent_frame, bg="#121212")
            scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Adiciona suporte para scroll com mouse wheel
            canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
            
            # Armazenar referências
            setattr(self, f"{categoria}_canvas", canvas)
            setattr(self, f"{categoria}_scrollbar", scrollbar)
            setattr(self, f"{categoria}_scrollable_frame", scrollable_frame)
            
            return scrollable_frame
        except Exception as e:
            print(f"Erro ao criar frame scrollable para {categoria}: {e}")
            return None
    
    def carregar_fotos_existentes(self):
        """Carrega as fotos existentes no diretório de histórico."""
        try:
            # Limpar listas existentes
            self.fotos_historico = []
            self.fotos_ok = []
            self.fotos_ng = []
            self.fotos_capturas = []
            self.programas_disponiveis = ["Todos"]
            
            # Função auxiliar para processar arquivos de uma pasta
            def processar_arquivos(diretorio, categoria):
                fotos = []
                if diretorio.exists():
                    for arquivo in diretorio.glob("*.png"):
                        try:
                            nome = arquivo.name
                            timestamp_str = ""
                            programa = "Desconhecido"
                            
                            # Extrair informações do nome do arquivo
                            if categoria == "capturas" and nome.startswith("foto_"):
                                # Formato: foto_modelo_YYYYMMDD_HHMMSS.png
                                partes = nome[5:-4].split('_')  # Remove "foto_" e ".png"
                                if len(partes) >= 2:
                                    # O último ou os dois últimos elementos são a data/hora
                                    if len(partes[-1]) == 6 and len(partes[-2]) == 8:  # HHMMSS e YYYYMMDD
                                        timestamp_str = f"{partes[-2]}_{partes[-1]}"
                                        programa = "_".join(partes[:-2]) if len(partes) > 2 else "Desconhecido"
                                    else:
                                        timestamp_str = partes[-1]
                                        programa = "_".join(partes[:-1]) if len(partes) > 1 else "Desconhecido"
                            elif (categoria == "ok" or categoria == "ng") and nome.startswith("inspecao_"):
                                # Formato: inspecao_modelo_YYYYMMDD_HHMMSS.png
                                partes = nome[9:-4].split('_')  # Remove "inspecao_" e ".png"
                                if len(partes) >= 2:
                                    # O último ou os dois últimos elementos são a data/hora
                                    if len(partes[-1]) == 6 and len(partes[-2]) == 8:  # HHMMSS e YYYYMMDD
                                        timestamp_str = f"{partes[-2]}_{partes[-1]}"
                                        programa = "_".join(partes[:-2]) if len(partes) > 2 else "Desconhecido"
                                    else:
                                        timestamp_str = partes[-1]
                                        programa = "_".join(partes[:-1]) if len(partes) > 1 else "Desconhecido"
                            
                            # Se encontrou um timestamp válido
                            if timestamp_str:
                                try:
                                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                    
                                    # Adicionar programa à lista de programas disponíveis
                                    if programa != "Desconhecido" and programa not in self.programas_disponiveis:
                                        self.programas_disponiveis.append(programa)
                                    
                                    foto_info = {
                                        'arquivo': arquivo,
                                        'timestamp': timestamp,
                                        'categoria': categoria,
                                        'programa': programa
                                    }
                                    fotos.append(foto_info)
                                except ValueError:
                                    print(f"Formato de timestamp inválido: {timestamp_str}")
                        except Exception as e:
                            print(f"Erro ao processar arquivo {arquivo}: {e}")
                return fotos
            
            # Processar arquivos de cada diretório
            self.fotos_ok = processar_arquivos(self.ok_dir, "ok")
            self.fotos_ng = processar_arquivos(self.ng_dir, "ng")
            self.fotos_capturas = processar_arquivos(self.capturas_dir, "capturas")
            
            # Combinar todas as fotos
            self.fotos_historico = self.fotos_ok + self.fotos_ng + self.fotos_capturas
            
            # Ordenar por timestamp (mais recente primeiro)
            self.fotos_historico.sort(key=lambda x: x['timestamp'], reverse=True)
            self.fotos_ok.sort(key=lambda x: x['timestamp'], reverse=True)
            self.fotos_ng.sort(key=lambda x: x['timestamp'], reverse=True)
            self.fotos_capturas.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Atualizar combobox de programas
            self.programa_combobox['values'] = self.programas_disponiveis
            self.programa_combobox.current(0)  # Selecionar "Todos"
        except Exception as e:
            print(f"Erro ao carregar fotos existentes: {e}")
    
    def exibir_fotos(self):
        """Exibe as fotos no histórico."""
        try:
            # Verificar se a interface foi inicializada
            if not hasattr(self, "todas_scrollable_frame") or self.todas_scrollable_frame is None:
                print("Interface não inicializada completamente. Tentando inicializar...")
                self.setup_ui()
                # Carregar fotos existentes
                self.carregar_fotos_existentes()
                # Atualizar combobox de programas
                if self.programa_combobox is not None:
                    self.programa_combobox['values'] = self.programas_disponiveis
                    self.programa_combobox.current(0)  # Selecionar "Todos"
                return
                
            # Obter programa selecionado
            programa = self.programa_selecionado.get()
            print(f"Filtrando por programa: {programa}")
            print(f"Programas disponíveis: {self.programas_disponiveis}")
            
            # Filtrar fotos por programa se necessário
            fotos_todas = [f for f in self.fotos_historico] if programa == "Todos" else \
                         [f for f in self.fotos_historico if f['programa'] == programa]
            fotos_ok = [f for f in self.fotos_ok] if programa == "Todos" else \
                      [f for f in self.fotos_ok if f['programa'] == programa]
            fotos_ng = [f for f in self.fotos_ng] if programa == "Todos" else \
                      [f for f in self.fotos_ng if f['programa'] == programa]
            fotos_capturas = [f for f in self.fotos_capturas] if programa == "Todos" else \
                           [f for f in self.fotos_capturas if f['programa'] == programa]
            
            print(f"Total de fotos filtradas: {len(fotos_todas)}")
            
            # Exibir fotos em cada aba
            self.exibir_fotos_em_aba(self.todas_scrollable_frame, fotos_todas, "todas")
            self.exibir_fotos_em_aba(self.ok_scrollable_frame, fotos_ok, "ok")
            self.exibir_fotos_em_aba(self.ng_scrollable_frame, fotos_ng, "ng")
            self.exibir_fotos_em_aba(self.capturas_scrollable_frame, fotos_capturas, "capturas")
        except Exception as e:
            print(f"Erro ao exibir fotos: {e}")
    
    def exibir_fotos_em_aba(self, frame, fotos, categoria):
        """Exibe as fotos em uma aba específica."""
        # Verificar se o frame existe
        if frame is None:
            print(f"Frame para categoria {categoria} não foi inicializado corretamente")
            return
            
        # Limpar frame existente
        for widget in frame.winfo_children():
            widget.destroy()
        
        if not fotos:
            # Mensagem quando não há fotos
            ttk.Label(frame, 
                     text="Nenhuma foto nesta categoria", 
                     font=("Arial", 12), 
                     foreground="#AAAAAA").pack(pady=20)
            return
        
        # Criar grid para exibir fotos (3 colunas)
        grid_frame = ttk.Frame(frame)
        grid_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Criar frames para cada coluna
        colunas = []
        for i in range(3):
            coluna = ttk.Frame(grid_frame)
            coluna.pack(side=LEFT, fill=BOTH, expand=True, padx=5)
            colunas.append(coluna)
        
        # Distribuir fotos pelas colunas
        for i, foto_info in enumerate(fotos):
            coluna_idx = i % 3
            try:
                if 'arquivo' in foto_info and foto_info['arquivo'].exists():
                    self.criar_card_foto(colunas[coluna_idx], foto_info)
                else:
                    print(f"Arquivo não encontrado para foto {i} na categoria {categoria}")
            except Exception as e:
                print(f"Erro ao criar card para foto {foto_info.get('arquivo', 'desconhecida')}: {e}")
                continue
    
    def filtrar_por_programa(self, event=None):
        """Filtra as fotos pelo programa selecionado."""
        try:
            programa = self.programa_selecionado.get()
            print(f"Filtro selecionado: {programa}")
            
            # Verificar se o programa está na lista de programas disponíveis
            if programa not in self.programas_disponiveis:
                print(f"Programa {programa} não encontrado na lista de programas disponíveis")
                return
                
            self.exibir_fotos()
        except Exception as e:
            print(f"Erro ao filtrar por programa: {e}")
    
    def criar_card_foto(self, parent_frame, foto_info):
        """Cria um card para exibir uma foto com suas informações."""
        try:
            # Frame para o card
            card_frame = ttk.Frame(parent_frame, relief="solid", borderwidth=1)
            card_frame.pack(fill=X, pady=10, padx=5)
            
            # Carregar e exibir a imagem
            img = cv2.imread(str(foto_info['arquivo']))
            if img is not None:
                # Redimensionar para exibição
                img_height, img_width = img.shape[:2]
                max_width = 300
                scale = max_width / img_width
                new_height = int(img_height * scale)
                img_resized = cv2.resize(img, (max_width, new_height))
                
                # Converter para formato Tkinter
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Label para a imagem
                img_label = ttk.Label(card_frame, image=img_tk)
                img_label.image = img_tk  # Manter referência
                img_label.pack(pady=5)
                
                # Informações da foto
                info_frame = ttk.Frame(card_frame)
                info_frame.pack(fill=X, padx=10, pady=5)
                
                # Data e hora
                timestamp = foto_info['timestamp']
                data_str = timestamp.strftime("%d/%m/%Y")
                hora_str = timestamp.strftime("%H:%M:%S")
                
                # Categoria e programa
                categoria = foto_info.get('categoria', 'desconhecida')
                programa = foto_info.get('programa', 'Desconhecido')
                
                # Cor baseada na categoria
                categoria_cor = "#00AA00" if categoria == "ok" else \
                               "#AA0000" if categoria == "ng" else \
                               "#0000AA" if categoria == "capturas" else "#AAAAAA"
                
                categoria_texto = "APROVADO" if categoria == "ok" else \
                                 "REPROVADO" if categoria == "ng" else \
                                 "CAPTURA MANUAL" if categoria == "capturas" else "DESCONHECIDO"
                
                ttk.Label(info_frame, text=f"📊 Status: {categoria_texto}", 
                         font=("Arial", 10, "bold"), foreground=categoria_cor).pack(anchor="w")
                ttk.Label(info_frame, text=f"🔧 Programa: {programa}", font=("Arial", 10)).pack(anchor="w")
                ttk.Label(info_frame, text=f"📅 Data: {data_str}", font=("Arial", 10)).pack(anchor="w")
                ttk.Label(info_frame, text=f"🕒 Hora: {hora_str}", font=("Arial", 10)).pack(anchor="w")
                ttk.Label(info_frame, text=f"📏 Dimensões: {img_width}x{img_height}", font=("Arial", 9)).pack(anchor="w")
                
                # Botões de ação
                btn_frame = ttk.Frame(card_frame)
                btn_frame.pack(fill=X, padx=10, pady=5)
                
                # Botão para visualizar em tamanho real
                btn_visualizar = ttk.Button(btn_frame, text="Visualizar", 
                                         command=lambda: self.visualizar_foto(foto_info))
                btn_visualizar.pack(side=LEFT, padx=5)
                
                # Botão para excluir
                btn_excluir = ttk.Button(btn_frame, text="Excluir", 
                                       command=lambda: self.excluir_foto(foto_info, card_frame))
                btn_excluir.pack(side=RIGHT, padx=5)
            else:
                ttk.Label(card_frame, text="Erro ao carregar imagem", foreground="red").pack(pady=10)
        
        except Exception as e:
            print(f"Erro ao criar card para foto: {e}")
    
    def visualizar_foto(self, foto_info):
        """Abre uma janela para visualizar a foto em tamanho real com zoom."""
        try:
            img = cv2.imread(str(foto_info['arquivo']))
            if img is not None:
                # Criar janela de visualização
                view_window = Toplevel(self)
                view_window.title(f"Foto - {foto_info['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
                
                # Ajustar tamanho da janela (máximo 80% da tela)
                screen_width = view_window.winfo_screenwidth()
                screen_height = view_window.winfo_screenheight()
                
                img_height, img_width = img.shape[:2]
                scale = min(0.8 * screen_width / img_width, 0.8 * screen_height / img_height)
                
                if scale < 1:  # Redimensionar apenas se for maior que 80% da tela
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                
                # Converter para formato Tkinter
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Canvas para exibir a imagem com scrollbars
                canvas_frame = ttk.Frame(view_window)
                canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
                
                h_scrollbar = ttk.Scrollbar(canvas_frame, orient=HORIZONTAL)
                h_scrollbar.pack(side=BOTTOM, fill=X)
                
                v_scrollbar = ttk.Scrollbar(canvas_frame, orient=VERTICAL)
                v_scrollbar.pack(side=RIGHT, fill=Y)
                
                canvas = Canvas(canvas_frame, 
                               xscrollcommand=h_scrollbar.set,
                               yscrollcommand=v_scrollbar.set)
                canvas.pack(side=LEFT, fill=BOTH, expand=True)
                
                h_scrollbar.config(command=canvas.xview)
                v_scrollbar.config(command=canvas.yview)
                
                # Exibir imagem no canvas
                canvas.create_image(0, 0, anchor=NW, image=img_tk)
                canvas.image = img_tk  # Manter referência
                
                # Configurar região de rolagem
                canvas.config(scrollregion=canvas.bbox("all"))
                
                # Variáveis para controle de zoom
                self.zoom_level = 1.0
                self.original_img = img_rgb
                self.current_img_tk = img_tk
                
                # Função para aplicar zoom
                def apply_zoom(event):
                    # Determinar direção do scroll
                    if event.delta > 0:
                        # Zoom in
                        self.zoom_level *= 1.1
                    else:
                        # Zoom out
                        self.zoom_level /= 1.1
                    
                    # Limitar zoom
                    self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
                    
                    # Calcular novas dimensões
                    new_width = int(img_width * self.zoom_level)
                    new_height = int(img_height * self.zoom_level)
                    
                    # Redimensionar imagem
                    img_resized = cv2.resize(self.original_img, (new_width, new_height))
                    img_pil = Image.fromarray(img_resized)
                    self.current_img_tk = ImageTk.PhotoImage(img_pil)
                    
                    # Atualizar canvas
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor=NW, image=self.current_img_tk)
                    canvas.image = self.current_img_tk  # Manter referência
                    
                    # Atualizar região de rolagem
                    canvas.config(scrollregion=canvas.bbox("all"))
                
                # Vincular evento de scroll do mouse para zoom
                canvas.bind("<MouseWheel>", apply_zoom)
                
                # Botão para fechar
                ttk.Button(view_window, text="Fechar", command=view_window.destroy).pack(pady=10)
            else:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
        except Exception as e:
            print(f"Erro ao visualizar foto: {e}")
            messagebox.showerror("Erro", f"Erro ao visualizar foto: {e}")
    
    def excluir_foto(self, foto_info, card_frame):
        """Exclui uma foto do histórico."""
        try:
            if messagebox.askyesno("Confirmar", "Deseja realmente excluir esta foto do histórico?"):
                # Excluir arquivo
                if foto_info['arquivo'].exists():
                    foto_info['arquivo'].unlink()
                
                # Remover da lista
                self.fotos_historico = [f for f in self.fotos_historico if f['arquivo'] != foto_info['arquivo']]
                
                # Remover card da interface
                card_frame.destroy()
                
                messagebox.showinfo("Sucesso", "Foto excluída com sucesso!")
        except Exception as e:
            print(f"Erro ao excluir foto: {e}")
            messagebox.showerror("Erro", f"Erro ao excluir foto: {e}")
    

    
    def start_background_frame_capture(self):
        """Inicia a captura contínua de frames em segundo plano."""
        def capture_frames():
            while self.live_capture and self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.latest_frame = frame.copy()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura de frame: {e}")
                    break
        
        import threading
        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()
    
    def atualizar_historico(self):
        """Atualiza a exibição do histórico de fotos."""
        self.carregar_fotos_existentes()
        self.exibir_fotos()
        messagebox.showinfo("Sucesso", "Histórico atualizado!")

    
    def limpar_historico(self):
        """Limpa todo o histórico de fotos."""
        try:
            if messagebox.askyesno("Confirmar", "Deseja realmente limpar todo o histórico de fotos? Esta ação não pode ser desfeita."):
                # Perguntar se deseja limpar todas as categorias ou apenas uma específica
                opcoes = ["Todas as categorias", "Apenas Aprovadas (OK)", "Apenas Reprovadas (NG)", "Apenas Capturas Manuais"]
                resposta = simpledialog.askstring(
                    "Selecionar categoria", 
                    "Qual categoria deseja limpar?", 
                    initialvalue=opcoes[0],
                    parent=self
                )
                
                if not resposta:
                    return  # Usuário cancelou
                
                # Determinar quais listas limpar
                limpar_ok = resposta == opcoes[0] or resposta == opcoes[1]
                limpar_ng = resposta == opcoes[0] or resposta == opcoes[2]
                limpar_capturas = resposta == opcoes[0] or resposta == opcoes[3]
                
                # Excluir arquivos das categorias selecionadas
                if limpar_ok:
                    for foto_info in self.fotos_ok:
                        if foto_info['arquivo'].exists():
                            foto_info['arquivo'].unlink()
                    self.fotos_ok = []
                
                if limpar_ng:
                    for foto_info in self.fotos_ng:
                        if foto_info['arquivo'].exists():
                            foto_info['arquivo'].unlink()
                    self.fotos_ng = []
                
                if limpar_capturas:
                    for foto_info in self.fotos_capturas:
                        if foto_info['arquivo'].exists():
                            foto_info['arquivo'].unlink()
                    self.fotos_capturas = []
                
                # Atualizar lista combinada
                self.fotos_historico = self.fotos_ok + self.fotos_ng + self.fotos_capturas
                
                # Atualizar interface
                self.exibir_fotos()
                
                messagebox.showinfo("Sucesso", "Histórico de fotos limpo com sucesso!")
        except Exception as e:
            print(f"Erro ao limpar histórico: {e}")
            messagebox.showerror("Erro", f"Erro ao limpar histórico: {e}")


def main():
    """Função principal do módulo montagem."""
    root = create_main_window()
    root.mainloop()
    return root


if __name__ == "__main__":
    main()