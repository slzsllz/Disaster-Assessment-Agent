-- 为 chat_messages 表添加 report_files 列（AI 生成的 PDF 报告二进制数据）
-- 幂等：如果列已存在则跳过
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chat_messages' AND column_name = 'report_files'
    ) THEN
        ALTER TABLE chat_messages ADD COLUMN report_files JSONB;
    END IF;
END $$;
