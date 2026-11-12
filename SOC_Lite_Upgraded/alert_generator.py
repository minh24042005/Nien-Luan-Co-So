import datetime


class SOCAlertGenerator:
    def __init__(self):
        # Ánh xạ nhóm tấn công sang Mức độ nghiêm trọng (Severity) và Khuyến nghị (Action)
        self.severity_map = {
            'Normal': ('INFO', 'ALLOW'),
            'DoS': ('HIGH', 'BLOCK_IP'),
            'Probe': ('MEDIUM', 'LOG_ONLY'),
            'R2L': ('CRITICAL', 'ISOLATE_HOST'),
            'U2R': ('CRITICAL', 'ISOLATE_HOST')
        }

    def create_alert(
        self, flow_id: str, pred_label: str, confidence: float, protocol: str, service: str
    ) -> dict:
        severity, action = self.severity_map.get(pred_label, ('MEDIUM', 'LOG_ONLY'))
        # Chuẩn hóa múi giờ Việt Nam (UTC+7) theo ISO-8601
        tz_vn = datetime.timezone(datetime.timedelta(hours=7))
        now_iso = datetime.datetime.now(tz_vn).isoformat(timespec='seconds')

        return {
            "flow_id": flow_id,
            "timestamp": now_iso,
            "protocol": str(protocol).upper(),
            "service": str(service),
            "prediction_label": "Attack" if pred_label != 'Normal' else "Normal",
            "attack_category": pred_label,
            "confidence_score": round(float(confidence), 4),
            "severity": severity,
            "recommended_action": action
        }