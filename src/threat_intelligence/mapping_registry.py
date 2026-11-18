"""Dataset Label to ATT&CK Technique Mapping Registry (CP-10).

Manages conservative, human-reviewable candidate associations between dataset
attack classes and MITRE ATT&CK Enterprise techniques:
  - Maintains persistent mapping catalog in data/knowledge/mappings/label_mappings.json
  - Enforces explicit human approval workflow (Candidate -> Reviewed -> Approved/Rejected)
  - Records an append-only audit trail in data/knowledge/mappings/mapping_history.jsonl
  - Strict Invariant: Mapping reviews NEVER mutate CP-08 predictions or CP-09 reviews
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

from config.settings import settings
from src.threat_intelligence.attack_models import (
    LabelTechniqueMapping,
    MappingReviewEvent,
    MappingStatus,
    MappingStrength,
)
from src.threat_intelligence.taxonomy_service import TaxonomyService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Standard baseline candidate mappings for CIC-IDS-2017 classes
DEFAULT_BASELINE_MAPPINGS: List[Dict[str, Any]] = [
    {
        "mapping_id": "map_T1046_PortScan",
        "original_dataset_label": "PortScan",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "domain": "Enterprise",
        "tactic": "Discovery",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.DIRECT,
        "mapping_rationale": "Direct behavioral alignment: systematic transmission of SYN/ACK probes across destination port ranges.",
        "required_evidence": ["Target port frequency", "Sequential destination ports", "Rapid flow inter-arrival times"],
        "available_evidence": ["Destination Port", "Flow Duration", "Flow Packets/s", "Total Fwd Packets"],
        "missing_evidence": ["Attacking process identity", "Target socket responses", "Forensic packet payload content"],
        "mitigation_ids": ["M1031", "M1037", "M1030"],
        "detection_strategy_notes": "Monitor network flows for abnormal volumes of connection attempts to unmapped or closed ports.",
    },
    {
        "mapping_id": "map_T1110_FTP_Patator",
        "original_dataset_label": "FTP-Patator",
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "domain": "Enterprise",
        "tactic": "Credential Access",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.CONDITIONAL,
        "mapping_rationale": "Automated dictionary attack attempting multiple authentication credentials against FTP services (TCP 21).",
        "required_evidence": ["Repeated connections to TCP port 21", "High frequency of short-duration flows"],
        "available_evidence": ["Destination Port (21)", "Flow IAT", "Fwd Packet Length"],
        "missing_evidence": ["FTP authentication failure status codes (530)", "Username/password dictionary strings"],
        "mitigation_ids": ["M1036", "M1032"],
        "detection_strategy_notes": "Correlate network flow rates with FTP daemon authentication logs.",
    },
    {
        "mapping_id": "map_T1110_SSH_Patator",
        "original_dataset_label": "SSH-Patator",
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "domain": "Enterprise",
        "tactic": "Credential Access",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.CONDITIONAL,
        "mapping_rationale": "Automated dictionary attack attempting repeated authentication handshakes against SSH services (TCP 22).",
        "required_evidence": ["Repeated connections to TCP port 22", "High connection frequency"],
        "available_evidence": ["Destination Port (22)", "Flow Duration", "Packet count"],
        "missing_evidence": ["SSH auth.log failure events", "Encrypted payload contents"],
        "mitigation_ids": ["M1036", "M1032"],
        "detection_strategy_notes": "Monitor for multiple failed SSH login handshakes from identical source IP addresses.",
    },
    {
        "mapping_id": "map_T1110_Web_Brute_Force",
        "original_dataset_label": "Web Attack - Brute Force",
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "domain": "Enterprise",
        "tactic": "Credential Access",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.CONDITIONAL,
        "mapping_rationale": "Automated credential stuffing targeting web application login interfaces over HTTP/HTTPS.",
        "required_evidence": ["Repeated POST requests to login URIs", "Rapid sequential HTTP transactions"],
        "available_evidence": ["Destination Port (80/443)", "Flow Bytes/s", "SYN flag count"],
        "missing_evidence": ["HTTP POST body credential parameters", "Web server 401/403 status responses"],
        "mitigation_ids": ["M1036", "M1032"],
        "detection_strategy_notes": "Analyze web access logs for spikes in authentication failure responses.",
    },
    {
        "mapping_id": "map_T1498_DoS_Hulk",
        "original_dataset_label": "DoS Hulk",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "domain": "Enterprise",
        "tactic": "Impact",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Volumetric HTTP GET flooding designed to exhaust web application server worker threads.",
        "required_evidence": ["Volumetric flow generation", "Asymmetric forward/backward packet counts"],
        "available_evidence": ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s"],
        "missing_evidence": ["Web server CPU/memory exhaustion metrics", "Service disruption confirmation"],
        "mitigation_ids": ["M1037", "M1031"],
        "detection_strategy_notes": "Monitor ingress bandwidth and HTTP request frequency per client IP.",
    },
    {
        "mapping_id": "map_T1498_DoS_GoldenEye",
        "original_dataset_label": "DoS GoldenEye",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "domain": "Enterprise",
        "tactic": "Impact",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Application-layer denial of service keeping socket connections open via Keep-Alive headers.",
        "required_evidence": ["Extended flow duration", "Multiple requests over persistent connections"],
        "available_evidence": ["Flow Duration", "ACK Flag Count", "PSH Flag Count"],
        "missing_evidence": ["HTTP Keep-Alive timeout metrics", "Server pool saturation telemetry"],
        "mitigation_ids": ["M1037", "M1031"],
        "detection_strategy_notes": "Inspect connection hold times and limit maximum concurrent persistent HTTP sessions.",
    },
    {
        "mapping_id": "map_T1498_DoS_slowloris",
        "original_dataset_label": "DoS slowloris",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "domain": "Enterprise",
        "tactic": "Impact",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Low-and-slow HTTP header starvation holding server connections open with periodic header updates.",
        "required_evidence": ["Abnormally long flow duration with very low packet volume"],
        "available_evidence": ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s (low)"],
        "missing_evidence": ["Partial HTTP header fragments", "Server connection table exhaustion state"],
        "mitigation_ids": ["M1037", "M1031"],
        "detection_strategy_notes": "Set aggressive HTTP header read timeouts on web servers and load balancers.",
    },
    {
        "mapping_id": "map_T1498_DoS_Slowhttptest",
        "original_dataset_label": "DoS Slowhttptest",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "domain": "Enterprise",
        "tactic": "Impact",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Slow read/slow body denial-of-service attack draining server connection pools.",
        "required_evidence": ["Prolonged flow duration with low transmission rates"],
        "available_evidence": ["Flow Duration", "Flow Bytes/s", "Fwd Packet Length Mean"],
        "missing_evidence": ["Application socket buffer fill rates", "Target server connection thread pool logs"],
        "mitigation_ids": ["M1037", "M1031"],
        "detection_strategy_notes": "Implement minimum incoming data transfer rate thresholds on edge proxies.",
    },
    {
        "mapping_id": "map_T1498_DDoS",
        "original_dataset_label": "DDoS",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "domain": "Enterprise",
        "tactic": "Impact",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Distributed high-volume packet flooding overwhelming network infrastructure capacity.",
        "required_evidence": ["Massive volumetric flow aggregate", "Multiple disparate source IPs"],
        "available_evidence": ["Flow Packets/s", "Total Fwd Packets", "Flow Duration"],
        "missing_evidence": ["Distributed source IP correlation", "Upstream transit pipe utilization metrics"],
        "mitigation_ids": ["M1037", "M1030"],
        "detection_strategy_notes": "Deploy automated BGP flowspec or upstream scrubbing centers upon bandwidth threshold breach.",
    },
    {
        "mapping_id": "map_T1190_SQL_Injection",
        "original_dataset_label": "Web Attack - Sql Injection",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "domain": "Enterprise",
        "tactic": "Initial Access",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.CONDITIONAL,
        "mapping_rationale": "Exploitation of web input fields to inject arbitrary SQL syntax into backend databases.",
        "required_evidence": ["HTTP GET/POST request with SQL keywords (UNION, SELECT, --)"],
        "available_evidence": ["Destination Port (80/443)", "Flow Duration", "Packet lengths"],
        "missing_evidence": ["Full unencrypted HTTP payload inspection", "Database query execution logs"],
        "mitigation_ids": ["M1031", "M1050"],
        "detection_strategy_notes": "Deploy Web Application Firewalls (WAF) inspecting SQL syntax in query parameters.",
    },
    {
        "mapping_id": "map_T1190_Heartbleed",
        "original_dataset_label": "Heartbleed",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "domain": "Enterprise",
        "tactic": "Initial Access",
        "mapping_status": MappingStatus.CANDIDATE,
        "mapping_strength": MappingStrength.BROAD,
        "mapping_rationale": "Exploitation of OpenSSL TLS Heartbeat memory disclosure vulnerability (CVE-2014-0160).",
        "required_evidence": ["Malformed TLS Heartbeat request with payload length discrepancy"],
        "available_evidence": ["Destination Port (443)", "Packet lengths", "Protocol (6 / TCP)"],
        "missing_evidence": ["Decrypted TLS record layer bytes", "Server memory buffer disclosure logs"],
        "mitigation_ids": ["M1051", "M1031"],
        "detection_strategy_notes": "Patch OpenSSL to non-vulnerable versions and inspect TLS records with deep packet inspection (DPI).",
    },
    {
        "mapping_id": "map_unmapped_Web_XSS",
        "original_dataset_label": "Web Attack - XSS",
        "technique_id": "",
        "technique_name": "Unmapped",
        "domain": "Enterprise",
        "tactic": "Initial Access / Execution",
        "mapping_status": MappingStatus.UNMAPPED,
        "mapping_strength": MappingStrength.INSUFFICIENT,
        "mapping_rationale": "XSS targets client browsers via web parameters; netflow telemetry lacks HTML/JS payload inspection to establish ATT&CK technique.",
        "required_evidence": ["Client-side script execution in victim browser"],
        "available_evidence": ["Destination Port (80/443)"],
        "missing_evidence": ["Full HTTP response body", "Client DOM execution events"],
        "mitigation_ids": ["M1050"],
        "detection_strategy_notes": "Deploy Content Security Policy (CSP) and input sanitization on web applications.",
    },
    {
        "mapping_id": "map_unmapped_Bot",
        "original_dataset_label": "Bot",
        "technique_id": "",
        "technique_name": "Unmapped",
        "domain": "Enterprise",
        "tactic": "Command and Control",
        "mapping_status": MappingStatus.UNMAPPED,
        "mapping_strength": MappingStrength.INSUFFICIENT,
        "mapping_rationale": "Broad label without sufficient flow-level evidence to isolate specific C2 protocol or technique.",
        "required_evidence": ["Identified C2 domain or communication channel"],
        "available_evidence": ["Flow Duration", "Packet lengths"],
        "missing_evidence": ["Process names", "DNS resolution logs", "C2 protocol signatures"],
        "mitigation_ids": [],
        "detection_strategy_notes": "Correlate host telemetry and DNS requests with known botnet indicator feeds.",
    },
    {
        "mapping_id": "map_unmapped_Infiltration",
        "original_dataset_label": "Infiltration",
        "technique_id": "",
        "technique_name": "Unmapped",
        "domain": "Enterprise",
        "tactic": "Lateral Movement / Execution",
        "mapping_status": MappingStatus.UNMAPPED,
        "mapping_strength": MappingStrength.INSUFFICIENT,
        "mapping_rationale": "Complex multi-stage intrusion; single network flow lacks context to isolate a specific technique.",
        "required_evidence": ["Host exploit execution", "Internal lateral movement telemetry"],
        "available_evidence": ["Flow metrics"],
        "missing_evidence": ["Host event logs", "Process execution hierarchy", "Command line arguments"],
        "mitigation_ids": [],
        "detection_strategy_notes": "Deploy Endpoint Detection & Response (EDR) to capture post-exploitation behaviors.",
    },
    {
        "mapping_id": "map_unmapped_BENIGN",
        "original_dataset_label": "BENIGN",
        "technique_id": "",
        "technique_name": "Unmapped",
        "domain": "Enterprise",
        "tactic": "None",
        "mapping_status": MappingStatus.UNMAPPED,
        "mapping_strength": MappingStrength.INSUFFICIENT,
        "mapping_rationale": "Normal benign traffic does not correspond to an adversary ATT&CK technique.",
        "required_evidence": [],
        "available_evidence": ["Normal network telemetry"],
        "missing_evidence": [],
        "mitigation_ids": [],
        "detection_strategy_notes": "Maintain normal network baselines.",
    },
]


class MappingRegistry:
    """Manages persistent label-to-technique mappings and human review audit journals."""

    _REGISTRY_LOCK = threading.RLock()

    def __init__(self, mappings_dir: Optional[Path] = None):
        """Initialize registry with mappings directory."""
        self.mappings_dir = mappings_dir or getattr(
            settings, "mappings_dir", settings.data_dir / "knowledge" / "mappings"
        )
        self.mappings_dir.mkdir(parents=True, exist_ok=True)
        self.mappings_file = self.mappings_dir / "label_mappings.json"
        self.history_file = self.mappings_dir / "mapping_history.jsonl"
        self._initialize_defaults_if_missing()

    def _initialize_defaults_if_missing(self) -> None:
        """Create initial conservative baseline mappings if no persistent file exists."""
        with self._REGISTRY_LOCK:
            if not self.mappings_file.exists():
                mappings: Dict[str, Dict[str, Any]] = {}
                for item in DEFAULT_BASELINE_MAPPINGS:
                    m = LabelTechniqueMapping(
                        mapping_id=item["mapping_id"],
                        original_dataset_label=item["original_dataset_label"],
                        normalized_dataset_label=TaxonomyService.normalize_label(item["original_dataset_label"]),
                        technique_id=item["technique_id"],
                        technique_name=item["technique_name"],
                        domain=item["domain"],
                        tactic=item["tactic"],
                        mapping_status=item["mapping_status"],
                        mapping_strength=item["mapping_strength"],
                        mapping_rationale=item["mapping_rationale"],
                        required_evidence=item["required_evidence"],
                        available_evidence=item["available_evidence"],
                        missing_evidence=item["missing_evidence"],
                        mitigation_ids=item["mitigation_ids"],
                        detection_strategy_notes=item["detection_strategy_notes"],
                        source_references=["MITRE ATT&CK Enterprise v20.1", "CIC-IDS-2017 Dataset Specification"],
                    )
                    mappings[m.mapping_id] = m.to_dict()

                temp_path = self.mappings_file.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(mappings, f, indent=2)
                os.replace(temp_path, self.mappings_file)
                logger.info(f"Initialized {len(mappings)} default baseline ATT&CK mappings in {self.mappings_file}")

    def list_mappings(self) -> List[LabelTechniqueMapping]:
        """Retrieve all active mappings."""
        with self._REGISTRY_LOCK:
            if not self.mappings_file.exists():
                return []
            try:
                with open(self.mappings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [LabelTechniqueMapping.from_dict(v) for v in data.values()]
            except Exception as e:
                logger.error(f"Failed to read label mappings file: {e}")
                return []

    def get_mapping_by_id(self, mapping_id: str) -> Optional[LabelTechniqueMapping]:
        """Lookup a mapping by its mapping ID."""
        mappings = self.list_mappings()
        for m in mappings:
            if m.mapping_id == mapping_id:
                return m
        return None

    def get_mapping_for_label(self, dataset_label: str) -> Optional[LabelTechniqueMapping]:
        """Lookup candidate mapping for an original or normalized dataset label."""
        norm_key = TaxonomyService.normalize_label(dataset_label)
        mappings = self.list_mappings()
        for m in mappings:
            if m.normalized_dataset_label == norm_key:
                return m
        return None

    def approve_mapping(
        self,
        mapping_id: str,
        reviewer_label: Optional[str] = None,
        analyst_notes: Optional[str] = None,
    ) -> LabelTechniqueMapping:
        """Approve a candidate mapping (explicit human action)."""
        return self._update_status(
            mapping_id=mapping_id,
            new_status=MappingStatus.APPROVED,
            reviewer_label=reviewer_label,
            analyst_notes=analyst_notes,
        )

    def reject_mapping(
        self,
        mapping_id: str,
        reviewer_label: Optional[str] = None,
        analyst_notes: Optional[str] = None,
    ) -> LabelTechniqueMapping:
        """Reject a candidate mapping (explicit human action)."""
        return self._update_status(
            mapping_id=mapping_id,
            new_status=MappingStatus.REJECTED,
            reviewer_label=reviewer_label,
            analyst_notes=analyst_notes,
        )

    def mark_unmapped(
        self,
        mapping_id: str,
        reviewer_label: Optional[str] = None,
        analyst_notes: Optional[str] = None,
    ) -> LabelTechniqueMapping:
        """Mark a mapping as Unmapped."""
        return self._update_status(
            mapping_id=mapping_id,
            new_status=MappingStatus.UNMAPPED,
            reviewer_label=reviewer_label,
            analyst_notes=analyst_notes,
        )

    def _update_status(
        self,
        mapping_id: str,
        new_status: str,
        reviewer_label: Optional[str] = None,
        analyst_notes: Optional[str] = None,
    ) -> LabelTechniqueMapping:
        """Update mapping status and record review audit event."""
        if not MappingStatus.is_valid(new_status):
            raise ValueError(f"Invalid mapping status: '{new_status}'")

        clean_notes = analyst_notes.strip() if analyst_notes else None
        if clean_notes and len(clean_notes) > 2000:
            raise ValueError("Analyst notes exceed 2000 characters limit.")

        clean_label = reviewer_label.strip() if reviewer_label else None
        if clean_label:
            clean_label = "".join(c for c in clean_label if c.isprintable())[:100]

        with self._REGISTRY_LOCK:
            mappings_dict: Dict[str, Dict[str, Any]] = {}
            if self.mappings_file.exists():
                with open(self.mappings_file, "r", encoding="utf-8") as f:
                    mappings_dict = json.load(f)

            if mapping_id not in mappings_dict:
                raise KeyError(f"Mapping ID '{mapping_id}' not found in registry.")

            current_data = mappings_dict[mapping_id]
            prev_status = current_data.get("mapping_status", MappingStatus.CANDIDATE)
            prev_tech = current_data.get("technique_id", "")

            # Update mapping
            now_str = datetime.now(timezone.utc).isoformat()
            current_data["mapping_status"] = new_status
            current_data["last_reviewed_at_utc"] = now_str
            if clean_label:
                current_data["reviewer_label"] = clean_label
            if clean_notes:
                current_data["analyst_notes"] = clean_notes

            updated_mapping = LabelTechniqueMapping.from_dict(current_data)
            mappings_dict[mapping_id] = updated_mapping.to_dict()

            # Save mappings atomically
            temp_path = self.mappings_file.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(mappings_dict, f, indent=2)
            os.replace(temp_path, self.mappings_file)

            # Record Review Event in append-only history
            last_event = self._get_last_history_event()
            prev_event_hash = last_event.event_hash if last_event else None
            event_id = f"map_rev_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            event = MappingReviewEvent(
                event_id=event_id,
                mapping_id=mapping_id,
                timestamp_utc=now_str,
                previous_status=prev_status,
                new_status=new_status,
                previous_technique_id=prev_tech,
                new_technique_id=updated_mapping.technique_id,
                analyst_notes=clean_notes,
                reviewer_label=clean_label,
                previous_event_hash=prev_event_hash,
            )

            with open(self.history_file, "a", encoding="utf-8") as hf:
                hf.write(json.dumps(event.to_dict()) + "\n")

            logger.info(f"Recorded mapping review event '{event_id}' on mapping '{mapping_id}': {prev_status} -> {new_status}")
            return updated_mapping

    def _get_last_history_event(self) -> Optional[MappingReviewEvent]:
        """Read the last review event from history file for hash chaining."""
        if not self.history_file.exists():
            return None
        last_ev = None
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_ev = MappingReviewEvent.from_dict(json.loads(line))
        except Exception:
            pass
        return last_ev

    def get_history(self, mapping_id: Optional[str] = None) -> List[MappingReviewEvent]:
        """Retrieve review history events, optionally filtered by mapping ID."""
        with self._REGISTRY_LOCK:
            if not self.history_file.exists():
                return []
            events: List[MappingReviewEvent] = []
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ev = MappingReviewEvent.from_dict(json.loads(line))
                            if mapping_id is None or ev.mapping_id == mapping_id:
                                events.append(ev)
            except Exception as e:
                logger.error(f"Failed to read mapping history: {e}")
            return events

    def verify_audit_integrity(self) -> Tuple[bool, List[str]]:
        """Verify cryptographic hash chain of mapping review history."""
        with self._REGISTRY_LOCK:
            if not self.history_file.exists():
                return True, []
            errors: List[str] = []
            prev_hash: Optional[str] = None

            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = MappingReviewEvent.from_dict(json.loads(line))
                        except Exception as e:
                            errors.append(f"Line {line_idx}: Corrupted JSON ({e})")
                            continue

                        if ev.previous_event_hash != prev_hash:
                            errors.append(
                                f"Line {line_idx} (Event {ev.event_id}): Hash chain broken. "
                                f"Expected '{prev_hash}', got '{ev.previous_event_hash}'"
                            )

                        if ev.compute_hash() != ev.event_hash:
                            errors.append(f"Line {line_idx} (Event {ev.event_id}): Self-hash mismatch.")

                        prev_hash = ev.event_hash
            except Exception as e:
                errors.append(f"Fatal error verifying history: {e}")

            return len(errors) == 0, errors
