"""
AI-Enhanced Telegram Bot for Price Action Trading Bot
Adds AI-specific commands and buttons to the existing Telegram bot
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from .ai_strategy import AIStrategy
from .auto_trainer import AutoTrainer

logger = logging.getLogger(__name__)

class AITelegramBot:
    """AI-enhanced Telegram bot with additional AI features"""
    
    def __init__(self, base_telegram_bot, ai_strategy: AIStrategy, auto_trainer: AutoTrainer):
        self.base_bot = base_telegram_bot
        self.ai_strategy = ai_strategy
        self.auto_trainer = auto_trainer
        
        # AI-specific state
        self.ai_training_in_progress = False
        self.ai_analysis_subscribers = set()
        
    def setup_ai_commands(self, application):
        """Setup AI-specific command handlers"""
        
        # AI Command Handlers
        application.add_handler(CommandHandler("ai_status", self._cmd_ai_status))
        application.add_handler(CommandHandler("ai_train", self._cmd_ai_train))
        application.add_handler(CommandHandler("ai_retrain", self._cmd_ai_retrain))
        application.add_handler(CommandHandler("ai_performance", self._cmd_ai_performance))
        application.add_handler(CommandHandler("ai_analyze", self._cmd_ai_analyze))
        application.add_handler(CommandHandler("ai_models", self._cmd_ai_models))
        application.add_handler(CommandHandler("ai_config", self._cmd_ai_config))
        application.add_handler(CommandHandler("ai_reset", self._cmd_ai_reset))
        application.add_handler(CommandHandler("ai_auto_train", self._cmd_ai_auto_train))
        
        # AI Callback Handlers
        application.add_handler(CallbackQueryHandler(self._on_ai_callback))
        
        logger.info("AI Telegram commands setup completed")
    
    def _build_ai_keyboard(self) -> ReplyKeyboardMarkup:
        """Build AI-enhanced keyboard"""
        keyboard_layout = [
            ["ℹ️ Info", "👤 Account"],
            ["📊 Positions", "📋 Orders"],
            ["🟢 Buy", "🔴 Sell"],
            ["▶️ Start Trade", "⏹️ End Trade"],
            ["📈 Performance", "🧾 History"],
            ["🔔 Alerts On/Off", "📰 News"],
            ["🧠 Analyze Now", "🤖 AI Status"],
            ["🤖 AI Train", "🤖 AI Performance"],
        ]
        return ReplyKeyboardMarkup(
            keyboard_layout,
            resize_keyboard=True,
            one_time_keyboard=False,
            is_persistent=True,
        )
    
    async def _cmd_ai_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI strategy status"""
        try:
            status = self.ai_strategy.get_strategy_info()
            performance = self.ai_strategy.get_model_performance()
            
            # Build status message
            message_lines = [
                "🤖 **AI Strategy Status**",
                "",
                f"**Training Status:** {'✅ Trained' if status['is_trained'] else '❌ Not Trained'}",
                f"**Enabled:** {'✅ Yes' if status['enabled'] else '❌ No'}",
                f"**Prediction Horizon:** {status['prediction_horizon']} periods",
                f"**Confidence Threshold:** {status['min_confidence_threshold']:.2f}",
                f"**Risk/Reward Ratio:** {status['risk_reward_ratio']:.1f}",
                "",
                "**Performance Metrics:**",
                f"• Total Predictions: {performance['prediction_accuracy']['total_predictions']}",
                f"• Accuracy: {performance['prediction_accuracy']['accuracy']:.3f}",
                f"• Last Confidence: {performance.get('last_prediction', {}).get('confidence', 0):.3f}",
                "",
                "**Available Models:**",
            ]
            
            if status['available_models']:
                for model in status['available_models']:
                    message_lines.append(f"• {model}")
            else:
                message_lines.append("• No models available")
            
            # Add auto-trainer status
            trainer_status = self.auto_trainer.get_training_status()
            message_lines.extend([
                "",
                "**Auto-Trainer:**",
                f"• Running: {'✅ Yes' if trainer_status['is_running'] else '❌ No'}",
                f"• Last Training: {trainer_status['last_training_time'] or 'Never'}",
                f"• Total Trainings: {trainer_status['training_stats']['total_trainings']}",
                f"• Success Rate: {(trainer_status['training_stats']['successful_trainings'] / max(trainer_status['training_stats']['total_trainings'], 1)) * 100:.1f}%"
            ])
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI status command: {e}")
            await update.message.reply_text(f"❌ Error getting AI status: {e}")
    
    async def _cmd_ai_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train AI models"""
        if self.ai_training_in_progress:
            await update.message.reply_text("🔄 AI training already in progress...")
            return
        
        try:
            # Get data periods parameter
            data_periods = 2000
            if context.args and len(context.args) > 0:
                try:
                    data_periods = int(context.args[0])
                    data_periods = max(500, min(data_periods, 10000))  # Limit between 500-10000
                except ValueError:
                    await update.message.reply_text("❌ Invalid data periods. Using default 2000.")
            
            await update.message.reply_text(f"🤖 Starting AI training with {data_periods} data points...")
            
            # Start training in background
            self.ai_training_in_progress = True
            
            # Run training in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._train_ai_models, data_periods)
            
            self.ai_training_in_progress = False
            
            if 'error' in result:
                await update.message.reply_text(f"❌ Training failed: {result['error']}")
            else:
                message_lines = [
                    "✅ **AI Training Completed**",
                    "",
                    f"**Status:** {result.get('status', 'Unknown')}",
                    f"**Samples:** {result.get('n_samples', 0)}",
                    f"**Features:** {result.get('n_features', 0)}",
                    f"**Models Trained:** {result.get('models_trained', 0)}",
                ]
                
                # Add training results
                training_results = result.get('training_results', {})
                if training_results:
                    message_lines.extend(["", "**Model Performance:**"])
                    for model_name, model_result in training_results.items():
                        if 'error' not in model_result:
                            val_score = model_result.get('val_score', 0)
                            cv_mean = model_result.get('cv_mean', 0)
                            message_lines.append(f"• {model_name}: {val_score:.3f} (CV: {cv_mean:.3f})")
                
                message = "\n".join(message_lines)
                await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI train command: {e}")
            self.ai_training_in_progress = False
            await update.message.reply_text(f"❌ Training error: {e}")
    
    def _train_ai_models(self, data_periods: int) -> Dict:
        """Train AI models (runs in executor)"""
        try:
            # Get historical data
            historical_data = self.auto_trainer._get_historical_data(data_periods)
            
            if historical_data is None:
                return {'error': 'Failed to get historical data'}
            
            # Train models
            result = self.ai_strategy.train_models(historical_data)
            return result
            
        except Exception as e:
            logger.error(f"Error in AI model training: {e}")
            return {'error': str(e)}
    
    async def _cmd_ai_retrain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retrain AI models"""
        try:
            await update.message.reply_text("🔄 Retraining AI models...")
            
            # Trigger manual retrain
            result = self.auto_trainer.manual_retrain()
            
            if 'error' in result:
                await update.message.reply_text(f"❌ Retraining failed: {result['error']}")
            else:
                await update.message.reply_text("✅ AI models retrained successfully!")
            
        except Exception as e:
            logger.error(f"Error in AI retrain command: {e}")
            await update.message.reply_text(f"❌ Retraining error: {e}")
    
    async def _cmd_ai_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI performance report"""
        try:
            # Get performance report
            report = self.auto_trainer.get_performance_report()
            
            if 'error' in report:
                await update.message.reply_text(f"❌ Error getting performance report: {report['error']}")
                return
            
            summary = report.get('summary', {})
            
            message_lines = [
                "📊 **AI Performance Report**",
                "",
                "**Training Summary:**",
                f"• Total Trainings: {summary.get('total_trainings', 0)}",
                f"• Success Rate: {summary.get('success_rate', 0):.1f}%",
                f"• Recent Success Rate: {summary.get('recent_success_rate', 0):.1f}%",
                f"• Best Accuracy: {summary.get('best_accuracy', 0):.3f}",
                f"• Current Accuracy: {summary.get('current_accuracy', 0):.3f}",
                f"• Trend: {summary.get('accuracy_trend', 'unknown')}",
            ]
            
            # Add recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                message_lines.extend(["", "**Recommendations:**"])
                for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                    message_lines.append(f"{i}. {rec}")
            
            # Add recent alerts
            alerts = report.get('performance_alerts', [])
            if alerts:
                message_lines.extend(["", "**Recent Alerts:**"])
                for alert in alerts[-3:]:  # Show last 3
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M')
                    message_lines.append(f"• {alert_time}: {alert['type']} ({alert['value']:.3f})")
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI performance command: {e}")
            await update.message.reply_text(f"❌ Error getting performance: {e}")
    
    async def _cmd_ai_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Perform AI analysis on current market"""
        try:
            # Get current market data
            session = self.base_bot._get_session(update.effective_chat.id)
            if not session:
                await update.message.reply_text("Please /login first.")
                return
            
            # Get current symbol data
            current_symbol = session.get_symbol()
            df = session.get_rates(current_symbol, 'M15', 200)
            
            if df is None or df.empty:
                await update.message.reply_text("❌ Unable to get market data")
                return
            
            # Perform AI analysis
            analysis = self.ai_strategy.analyze(df)
            
            if analysis['signal'] == 'error':
                await update.message.reply_text(f"❌ AI analysis error: {analysis['reason']}")
                return
            
            # Build analysis message
            message_lines = [
                f"🤖 **AI Analysis - {current_symbol}**",
                "",
                f"**Signal:** {analysis['signal'].upper()}",
                f"**Confidence:** {analysis['confidence']:.3f}",
                f"**AI Prediction:** {analysis.get('ai_prediction', 'N/A')}",
                f"**Model Accuracy:** {analysis.get('model_accuracy', 0):.3f}",
            ]
            
            if analysis['signal'] in ['buy', 'sell']:
                message_lines.extend([
                    "",
                    "**Trade Parameters:**",
                    f"• Entry: {analysis['entry_price']:.5f}",
                    f"• Stop Loss: {analysis['stop_loss']:.5f}",
                    f"• Take Profit: {analysis['take_profit']:.5f}",
                ])
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI analyze command: {e}")
            await update.message.reply_text(f"❌ AI analysis error: {e}")
    
    async def _cmd_ai_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show available AI models"""
        try:
            model_performance = self.ai_strategy.model_manager.get_model_performance()
            
            message_lines = [
                "🤖 **AI Models Status**",
                "",
            ]
            
            if not model_performance:
                message_lines.append("No models available")
            else:
                for model_name, metadata in model_performance.items():
                    if isinstance(metadata, dict) and 'val_score' in metadata:
                        message_lines.extend([
                            f"**{model_name.upper()}:**",
                            f"• Validation Score: {metadata['val_score']:.3f}",
                            f"• CV Score: {metadata.get('cv_mean', 0):.3f} ± {metadata.get('cv_std', 0):.3f}",
                            f"• Training Time: {metadata.get('training_time', 0):.1f}s",
                            f"• Created: {metadata.get('created_at', 'Unknown')[:19]}",
                            ""
                        ])
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI models command: {e}")
            await update.message.reply_text(f"❌ Error getting models: {e}")
    
    async def _cmd_ai_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI configuration"""
        try:
            status = self.ai_strategy.get_strategy_info()
            trainer_status = self.auto_trainer.get_training_status()
            
            message_lines = [
                "⚙️ **AI Configuration**",
                "",
                "**Strategy Parameters:**",
                f"• Prediction Horizon: {status['prediction_horizon']}",
                f"• Confidence Threshold: {status['min_confidence_threshold']:.2f}",
                f"• Risk/Reward Ratio: {status['risk_reward_ratio']:.1f}",
                f"• ATR Multiplier: {status['atr_multiplier']:.1f}",
                "",
                "**Auto-Trainer Settings:**",
                f"• Auto Retrain: {'✅ Enabled' if trainer_status['auto_retrain_enabled'] else '❌ Disabled'}",
                f"• Performance Based: {'✅ Enabled' if trainer_status['performance_based_retrain'] else '❌ Disabled'}",
                f"• Time Based: {'✅ Enabled' if trainer_status['time_based_retrain'] else '❌ Disabled'}",
                f"• Retrain Interval: {trainer_status['retrain_interval_hours']} hours",
                f"• Performance Threshold: {self.auto_trainer.performance_threshold:.2f}",
            ]
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI config command: {e}")
            await update.message.reply_text(f"❌ Error getting config: {e}")
    
    async def _cmd_ai_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset AI models"""
        try:
            # Create confirmation keyboard
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Confirm Reset", callback_data="ai_reset_confirm"),
                    InlineKeyboardButton("❌ Cancel", callback_data="ai_reset_cancel")
                ]
            ])
            
            await update.message.reply_text(
                "⚠️ **Reset AI Models**\n\nThis will delete all trained models and reset the AI system. Are you sure?",
                parse_mode='Markdown',
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"Error in AI reset command: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def _cmd_ai_auto_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle auto-training"""
        try:
            trainer_status = self.auto_trainer.get_training_status()
            
            if trainer_status['is_running']:
                self.auto_trainer.stop_auto_training()
                await update.message.reply_text("🛑 Auto-training stopped")
            else:
                self.auto_trainer.start_auto_training()
                await update.message.reply_text("▶️ Auto-training started")
            
        except Exception as e:
            logger.error(f"Error in AI auto-train command: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def _on_ai_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle AI callback queries"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "ai_reset_confirm":
                # Reset AI models
                self.ai_strategy.reset_models()
                await query.message.reply_text("✅ AI models reset successfully")
                
            elif query.data == "ai_reset_cancel":
                await query.message.reply_text("❌ AI reset cancelled")
            
        except Exception as e:
            logger.error(f"Error in AI callback: {e}")
            await query.message.reply_text(f"❌ Error: {e}")
    
    def get_ai_keyboard(self) -> ReplyKeyboardMarkup:
        """Get AI-enhanced keyboard"""
        return self._build_ai_keyboard()
    
    def notify_ai_subscribers(self, message: str):
        """Notify AI analysis subscribers"""
        try:
            for chat_id in self.ai_analysis_subscribers:
                self.base_bot.notify(chat_id, message)
        except Exception as e:
            logger.error(f"Error notifying AI subscribers: {e}")
    
    def subscribe_ai_analysis(self, chat_id: int):
        """Subscribe to AI analysis updates"""
        self.ai_analysis_subscribers.add(chat_id)
        logger.info(f"Chat {chat_id} subscribed to AI analysis")
    
    def unsubscribe_ai_analysis(self, chat_id: int):
        """Unsubscribe from AI analysis updates"""
        self.ai_analysis_subscribers.discard(chat_id)
        logger.info(f"Chat {chat_id} unsubscribed from AI analysis")
