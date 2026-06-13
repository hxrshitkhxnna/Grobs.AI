from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
import time
from datetime import datetime
from app.database.session import get_db
from sqlalchemy.orm import Session
from app.services.evaluation_service import EvaluationService
import logging

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])
logger = logging.getLogger(__name__)

@router.post("/run")
async def run_evaluation(method: str = "heuristic", calibrate: bool = True, db: Session = Depends(get_db)):
    try:
        service = EvaluationService(db)
        results = await service.run_full_evaluation(method, calibrate)
        
        # Log results to terminal for project evaluator visibility
        print("\n" + "="*80)
        print(f" GROBSAI - SYSTEM EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"{'Feature Category':<30} | {'Compl.':<6} | {'Acc.':<6} | {'Prec.':<6} | {'Lat.':<6}")
        print("-" * 80)
        for f in results["features_data"]:
            print(f"{f['name']:<30} | {f['completeness']:>5}% | {f['accuracy']:>5}% | {f['precision']:>5}% | {f['efficiency']:>4}ms")
        print("-" * 80)
        print(f" OVERALL ACCURACY: {results['overall_accuracy']:>5}% | AVG LATENCY: {results['average_latency']:>5}ms")
        print("="*80 + "\n")
        
        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
