import time
import ccxt
import numpy as np
import requests
import threading
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random

# CONFIGURAÇÕES DA BINANCE
api_key = 'qD5ji53CwsE7TnVSn1VLuYHtCEKap5zs2Rf27Ctm7DLVIcQBuPPX5exbtfJr6lCt'
api_secret = 'BthePAEuTDu9V1va2ntovK05LWVhSkAabPLVTtWj7tUO3foJqKlJNxeaGDw7ibcd'
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True}  # <-- Adicione esta linha
})

# CONFIGURAÇÃO TELEGRAM
telegram_token = '7840391700:AAG2_ys3_1nlvTIoEqanxXH1w54H0atc7Z4'
chat_id = '1116255258'
def send_telegram(message):
    url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
    data = {'chat_id': chat_id, 'text': message}
    requests.post(url, data=data)

# FUNÇÃO PARA OBTER COTAÇÃO BRL/USDT
def get_brl_usdt_rate():
    ticker = exchange.fetch_ticker('USDT/BRL')
    rate = ticker['last']
    print(f'[INFO] Cotação USDT/BRL: {rate}')
    send_telegram(f'[INFO] Cotação USDT/BRL: {rate}')
    return rate

# PARES DE MERCADO
symbols = ['BTC/USDT', 'PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT', 'FLOKI/USDT']
fixed_brl = 25
brl_usdt_rate = get_brl_usdt_rate()
fixed_usdt = fixed_brl / brl_usdt_rate
print(f'[INFO] Valor base em USDT: {fixed_usdt}')
send_telegram(f'[INFO] Valor base em USDT: {fixed_usdt}')

# TAKE PROFIT (ex: 10% de lucro)
take_profit_pct = 0.15
stop_loss_pct = 0.10  # 10% de stop loss
trailing_pct = 0.05   # trailing stop de 5%

# PREPARAR MODELOS POR PAR
models = {symbol: LogisticRegression() for symbol in symbols}
entry_prices = {symbol: None for symbol in symbols}
trailing_high = {symbol: None for symbol in symbols}

# VERIFICAR SALDO
def get_balance(asset):
    balance = exchange.fetch_balance()
    return balance.get(asset, {}).get('free', 0)
  
# COMPRAR FUNÇÃO
def buy(symbol, amount_usdt):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['ask']
    amount = amount_usdt / price
    order = exchange.create_market_buy_order(symbol, amount)
    send_telegram(f'COMPRADO: {amount:.6f} {symbol.split("/")[0]} @ {price}')
    return order, price

# VENDER FUNÇÃO
def sell(symbol, amount):
    order = exchange.create_market_sell_order(symbol, amount)
    send_telegram(f'VENDIDO: {amount:.6f} {symbol.split("/")[0]}')
    return order

# OBTER INDICADORES
def get_indicators(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    closes = df['close']
    volumes = df['volume']

    # RSI
    deltas = closes.diff()
    gain = deltas[deltas > 0].mean() if not deltas[deltas > 0].empty else 0
    loss = -deltas[deltas < 0].mean() if not deltas[deltas < 0].empty else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Médias móveis
    ma11 = closes.rolling(window=11).mean().iloc[-1]
    ma19 = closes.rolling(window=19).mean().iloc[-1]

    # MACD
    ema13 = closes.ewm(span=13, adjust=False).mean()
    ema24 = closes.ewm(span=24, adjust=False).mean()
    macd = ema13 - ema24
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_value = macd.iloc[-1]
    signal_value = signal.iloc[-1]

    # Volume médio
    avg_volume = volumes.rolling(window=20).mean().iloc[-1]

    return rsi, ma11, ma19, macd_value, signal_value, avg_volume

# PREDIÇÃO COM MELHORIA (USANDO RSI SIMPLES)
def ai_predict(symbol):
    rsi, ma11, ma19, macd_value, signal_value, avg_volume = get_indicators(symbol)
    return rsi

# FUNÇÃO PARA OBTER VOLATILIDADE
def get_volatility(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=30)
    closes = [c[4] for c in ohlcv]
    return np.std(closes) / np.mean(closes)

# LOOP PARA CADA PAR
def trade_loop(symbol):
    global entry_prices
    first_run = True
    last_usdt_balance = None  # Adicionado para controle do saldo
    while True:
        try:
            rsi, ma11, ma19, macd, signal, avg_volume = get_indicators(symbol)
            decision = 'hold'
            # Verifica se os indicadores não são NaN
            if not any(pd.isna(x) for x in [rsi, ma11, ma19, macd, signal]):
                # Exemplo de lógica combinada
                if rsi < 37 and macd > signal and ma11 > ma19:
                    decision = 'buy'
                elif rsi > 62 and macd < signal and ma11 < ma19:
                    decision = 'sell'

            usdt_balance = get_balance('USDT')
            # Mostra saldo apenas no início ou quando houver alteração
            if first_run or last_usdt_balance != usdt_balance:
                print(f'{symbol} -> Saldo USDT atual: {usdt_balance:.6f}')
                send_telegram(f'{symbol} -> Saldo USDT atual: {usdt_balance:.6f}')
                last_usdt_balance = usdt_balance
            if first_run:
                print(f'{symbol} -> RSI inicial: {rsi:.2f}')
                send_telegram(f'{symbol} -> RSI inicial: {rsi:.2f}')
                first_run = False

            usdt_balance = usdt_balance / len(symbols)
            vol = get_volatility(symbol)
            dynamic_usdt = min(fixed_usdt * (1 + vol), usdt_balance)
            
            if decision == 'buy' and entry_prices[symbol] is None:
                if usdt_balance >= dynamic_usdt:
                    order, price = buy(symbol, dynamic_usdt)
                    entry_prices[symbol] = price
                    send_telegram(f'{symbol} -> RSI no momento da compra: {rsi:.2f}')
                else:
                    print(f'Saldo USDT insuficiente para {symbol}')

            elif entry_prices[symbol]:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['bid']
                # Trailing stop
                if trailing_high[symbol] is None or current_price > trailing_high[symbol]:
                    trailing_high[symbol] = current_price
                stop_price = trailing_high[symbol] * (1 - trailing_pct)
                # Take profit
                if current_price >= entry_prices[symbol] * (1 + take_profit_pct):
                    asset = symbol.split('/')[0]
                    asset_balance = get_balance(asset)
                    if asset_balance > 0:
                        sell(symbol, asset_balance)
                        lucro = (current_price - entry_prices[symbol]) * asset_balance
                        lucro_pct = ((current_price / entry_prices[symbol]) - 1) * 100
                        msg = (f'{symbol} -> Lucro na venda: {lucro:.6f} USDT '
                               f'({lucro_pct:.2f}%)')
                        print(msg)
                        send_telegram(msg)
                        entry_prices[symbol] = None
                        trailing_high[symbol] = None
                # Stop loss
                elif current_price <= entry_prices[symbol] * (1 - stop_loss_pct):
                    asset = symbol.split('/')[0]
                    asset_balance = get_balance(asset)
                    if asset_balance > 0:
                        sell(symbol, asset_balance)
                        send_telegram(f'{symbol} -> Stop loss acionado!')
                        entry_prices[symbol] = None
                        trailing_high[symbol] = None
                # Trailing stop
                elif current_price <= stop_price and trailing_high[symbol] is not None:
                    asset = symbol.split('/')[0]
                    asset_balance = get_balance(asset)
                    if asset_balance > 0:
                        sell(symbol, asset_balance)
                        send_telegram(f'{symbol} -> Trailing stop acionado!')
                        entry_prices[symbol] = None
                        trailing_high[symbol] = None

            time.sleep(5)

        except ccxt.NetworkError as e:
            print(f'Erro de rede: {e}')
            send_telegram(f'Erro de rede: {e}')
            time.sleep(random.randint(10, 30))
        except ccxt.ExchangeError as e:
            print(f'Erro da exchange: {e}')
            send_telegram(f'Erro da exchange: {e}')
            time.sleep(random.randint(30, 60))
        except Exception as e:
            print(f'Erro no loop {symbol}: {e}')
            send_telegram(f'Erro no loop {symbol}: {e}')
            time.sleep(10)

# INICIAR THREADS
threads = []
for symbol in symbols:
    t = threading.Thread(target=trade_loop, args=(symbol,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
