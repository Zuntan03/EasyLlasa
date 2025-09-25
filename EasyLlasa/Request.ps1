param([string]$pathsFile)

# UTF-8エンコーディングでコンソール出力を設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Help display
if ($pathsFile -eq "--help" -or $pathsFile -eq "-h" -or $pathsFile -eq "/?" -or [string]::IsNullOrEmpty($pathsFile)) {
	Write-Host ""
	Write-Host "==============================================="
	Write-Host "Llasa Server Client Mode / Llasa サーバークライアントモード"
	Write-Host "==============================================="
	Write-Host ""
	Write-Host "Usage / 使用方法:"
	Write-Host "  Drag and drop audio files or folders to Generate.bat"
	Write-Host "  音声ファイルまたはフォルダをGenerate.batにドラッグ&ドロップ"
	Write-Host ""
	Write-Host "What it does / 動作内容:"
	Write-Host "  1. Takes your audio file(s) as voice reference"
	Write-Host "     音声ファイルを声の参照として使用"
	Write-Host "  2. Sends request to running Llasa server"
	Write-Host "     実行中のLlasaサーバーにリクエストを送信"
	Write-Host "  3. Server generates audio for each text line in Dialogue.txt"
	Write-Host "     サーバーがDialogue.txtの各テキスト行に対して音声生成"
	Write-Host "  4. Generated files are saved to Output folder"
	Write-Host "     生成されたファイルをOutputフォルダに保存"
	Write-Host ""
	Write-Host "Prerequisites / 前提条件:"
	Write-Host "  - StartServer.bat must be running first"
	Write-Host "    StartServer.batが先に実行されている必要があります"
	Write-Host "  - Dialogue.txt should contain the text lines to generate"
	Write-Host "    Dialogue.txtに生成したいテキスト行が含まれている必要があります"
	Write-Host ""
	Write-Host "Examples / 例:"
	Write-Host "  - Drag sample_voice.wav to Generate.bat"
	Write-Host "    sample_voice.wavをGenerate.batにドラッグ"
	Write-Host "  - Drag audio folder containing multiple files"
	Write-Host "    複数ファイルを含むフォルダをドラッグ"
	Write-Host "  - Drag multiple files at once"
	Write-Host "    複数ファイルを一度にドラッグ"
	Write-Host ""
	Write-Host "Command line usage / コマンドライン使用:"
	Write-Host "  Generate.bat `"C:\path\to\voice.wav`""
	Write-Host "  Generate.bat `"C:\path\to\audio_folder`""
	Write-Host "  Generate.bat `"voice1.wav`" `"voice2.wav`" `"folder`""
	Write-Host ""
	Write-Host "File flow / ファイルの流れ:"
	Write-Host "  Input: Audio files → Server: Processing → Output: Generated audio files"
	Write-Host "  入力: 音声ファイル → サーバー: 処理 → 出力: 生成された音声ファイル"
	Write-Host ""
	Write-Host "Supported formats / 対応形式: WAV, FLAC, OGG, MP3, AIFF, AU, CAF"
	Write-Host "Output location / 出力先: Output フォルダ"
	Write-Host ""
	Write-Host "Note / 注意:"
	Write-Host "  If StartServer.bat is not running, this will show an error."
	Write-Host "  StartServer.batが実行されていない場合、エラーが表示されます。"
	Write-Host ""
	$null = Read-Host "Press Enter to continue / Enterキーを押して続行"
	exit 1
}

Write-Host "PowerShell: Attempting to connect to named pipe 'llasa_pipe'..."
Write-Host "PowerShell: Reading paths from file: $pathsFile"

try {
	# 名前付きパイプクライアントを作成（双方向通信）
	$pipe = New-Object System.IO.Pipes.NamedPipeClientStream('.', 'llasa_pipe', [System.IO.Pipes.PipeDirection]::InOut)

	Write-Host "PowerShell: Connecting to pipe (timeout: 60 seconds)..."
	$pipe.Connect(60000)  # 60秒タイムアウト

	Write-Host "PowerShell: Connected successfully, sending data..."

	# パスファイルからパスを読み取り
	$pathArray = Get-Content $pathsFile -Encoding UTF8 | Where-Object { $_.Trim() -ne "" }
	
	$dataToSend = ($pathArray -join "`n") + "`n"
	Write-Host "PowerShell: Sending $($pathArray.Count) paths (line by line):"
	$pathArray | ForEach-Object { Write-Host "  '$_'" }
	
	# UTF-8エンコーディングでデータを送信
	$data = [System.Text.Encoding]::UTF8.GetBytes($dataToSend)
	$pipe.Write($data, 0, $data.Length)
	$pipe.Flush()

	Write-Host "PowerShell: Data sent successfully"

	# パイプを閉じる
	$pipe.Close()

	Write-Host "PowerShell: Connection closed successfully"
	exit 0

}
catch [System.TimeoutException] {
	Write-Host "PowerShell: Timeout - Server may not be running or pipe name is incorrect"
	exit 1
}
catch [System.IO.IOException] {
	Write-Host "PowerShell: IO Error - $($_.Exception.Message)"
	exit 1
}
catch {
	Write-Host "PowerShell: Unexpected error - $($_.Exception.Message)"
	exit 1
}
finally {
	# 確実にリソースを解放
	if ($pipe) { 
		try { $pipe.Dispose() } catch { }
	}
	# パスファイルをクリーンアップ
	if (Test-Path $pathsFile) {
		try { Remove-Item $pathsFile -Force } catch { }
	}
}