#!/usr/bin/env python3
"""
Industrial Multi-Stage Recommendation System Pipeline
Main Entry Point

Usage:
    python main.py --config config/pipeline_config.yaml --mode full
    python main.py --config config/pipeline_config.yaml --mode retrieve
    python main.py --config config/pipeline_config.yaml --mode personalize
    python main.py --config config/pipeline_config.yaml --mode rank
    python main.py --config config/pipeline_config.yaml --mode rerank
    python main.py --config config/pipeline_config.yaml --mode evaluate
    python main.py --config config/pipeline_config.yaml --mode serve --user_ids "U001,U002"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, StageLogger
from src.data_loader import DatasetBuilder, FeatureMapper, RecBoleConverter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Industrial Multi-Stage Recommendation System Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/pipeline_config.yaml',
        help='Path to configuration file (default: config/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'retrieve', 'personalize', 'rank', 'rerank', 'evaluate', 'serve', 'prepare'],
        default='full',
        help='Pipeline execution mode'
    )
    
    parser.add_argument(
        '--user_ids', '-u',
        type=str,
        default=None,
        help='Comma-separated user IDs for serving mode'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def run_pipeline(config: ConfigLoader, mode: str, args):
    """Run the recommendation pipeline."""
    
    logger = setup_logger(
        name='recsys_pipeline',
        log_dir=config.get('data.log_dir', 'logs'),
        level='DEBUG' if args.debug else config.get('logging.level', 'INFO')
    )
    
    logger.info("=" * 60)
    logger.info("Industrial Multi-Stage Recommendation System Pipeline")
    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {config.config_path}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        raise ValueError(f"Configuration validation failed: {errors}")
    
    output_dir = args.output_dir or config.get('data.output_dir', 'outputs')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Stage 0: Data Preparation
        if mode in ['full', 'prepare']:
            with StageLogger(logger, "Data Preparation"):
                builder = DatasetBuilder(config, logger)
                builder.load_data()
                builder.detect_fields()
                builder.preprocess_interactions()
                
                mapper = FeatureMapper(config, logger)
                interactions, user_features, item_features = builder.merge_features()
                
                lightgcn_data = mapper.map_for_lightgcn(interactions, user_features, item_features)
                sasrec_data = mapper.map_for_sasrec(interactions)
                deepfm_data = mapper.map_for_deepfm(interactions, user_features, item_features)
                
                converter = RecBoleConverter(config, logger)
                inter_dir = config.get('data.inter_dir', 'data/inter')
                dataset_config = converter.convert_interactions(interactions, inter_dir)
                
                results['data_prep'] = {
                    'n_interactions': len(interactions),
                    'n_users': interactions['user_id'].nunique(),
                    'n_items': interactions['item_id'].nunique(),
                    'dataset_config': dataset_config
                }
        
        if mode == 'prepare':
            logger.info("Data preparation complete. Run with --mode full for complete pipeline.")
            return results
        
        # Stage 1: LightGCN Retrieval
        if mode in ['full', 'retrieve']:
            with StageLogger(logger, "LightGCN Retrieval"):
                from src.retrieval.lightgnn_retriever import LightGCNRetriever
                
                retriever = LightGCNRetriever(config, logger)
                retrieval_results = retriever.run(lightgcn_data['interactions'])
                results['retrieval'] = retrieval_results
        
        # Stage 2: SASRec Personalization
        if mode in ['full', 'personalize']:
            with StageLogger(logger, "SASRec Personalization"):
                from src.sequential.sasrec_personalizer import SASRecPersonalizer
                
                personalizer = SASRecPersonalizer(config, logger)
                personalized_results = personalizer.run(sasrec_data)
                results['personalization'] = personalized_results
        
        # Stage 3: DeepFM Ranking
        if mode in ['full', 'rank']:
            with StageLogger(logger, "DeepFM Ranking"):
                from src.ranking.deepfm_ranker import DeepFMRanker
                
                ranker = DeepFMRanker(config, logger)
                ranking_results = ranker.run(deepfm_data)
                results['ranking'] = ranking_results
        
        # Stage 4: Diversity Re-ranking
        if mode in ['full', 'rerank']:
            with StageLogger(logger, "Diversity Re-ranking"):
                from src.reranking.diversity_reranker import DiversityReranker
                
                reranker = DiversityReranker(config, logger)
                rerank_results = reranker.run(ranking_results)
                results['reranking'] = rerank_results
        
        # Evaluation
        if mode in ['full', 'evaluate']:
            with StageLogger(logger, "Evaluation"):
                from src.evaluation.metrics_calculator import MetricsCalculator
                
                evaluator = MetricsCalculator(config, logger)
                eval_results = evaluator.evaluate(results)
                results['evaluation'] = eval_results
        
        # Save results
        from src.utils.helpers import save_json, save_parquet
        
        if mode in ['full', 'rerank', 'serve']:
            # Save final recommendations
            if 'reranking' in results and 'final_recommendations' in results['reranking']:
                rec_df = results['reranking']['final_recommendations']
                save_parquet(rec_df, Path(output_dir) / 'top_k_recommendations.parquet')
                logger.info(f"Saved final recommendations to {output_dir}/top_k_recommendations.parquet")
            
            # Save evaluation metrics
            if 'evaluation' in results:
                save_json(results['evaluation'], Path(output_dir) / 'evaluation_metrics.json')
                logger.info(f"Saved evaluation metrics to {output_dir}/evaluation_metrics.json")
            
            # Save pipeline summary
            summary = {
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'results_summary': {
                    stage: {k: v for k, v in data.items() if not isinstance(v, dict)}
                    for stage, data in results.items()
                }
            }
            save_json(summary, Path(output_dir) / 'pipeline_summary.json')
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


def main():
    args = parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    config.load()
    
    # Override output dir if specified
    if args.output_dir:
        config.set('data.output_dir', args.output_dir)
    
    # Run pipeline
    results = run_pipeline(config, args.mode, args)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    if 'data_prep' in results:
        dp = results['data_prep']
        print(f"Data: {dp['n_interactions']} interactions, "
              f"{dp['n_users']} users, {dp['n_items']} items")
    
    if 'retrieval' in results:
        print(f"Retrieval: {results['retrieval'].get('n_candidates', 0)} candidates generated")
    
    if 'evaluation' in results:
        eval_res = results['evaluation']
        if 'ranking_metrics' in eval_res:
            print(f"Evaluation: NDCG@10={eval_res['ranking_metrics'].get('NDCG@10', 'N/A')}")
    
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
