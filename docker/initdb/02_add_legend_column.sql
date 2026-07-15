-- 为已有的 chat_messages 表添加 legend 列（图例数据）
-- 幂等：如果列已存在则跳过
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chat_messages' AND column_name = 'legend'
    ) THEN
        ALTER TABLE chat_messages ADD COLUMN legend JSONB;
    END IF;
END $$;
