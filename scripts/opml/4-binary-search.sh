export ID=0
for i in {1..25}; do
    echo ""
    echo "--- STEP $i / 25 ---"
    echo ""
    BASEDIR=/tmp/cannon_fault CHALLENGER=1 npx hardhat run scripts/respond.js --network localhost
    BASEDIR=/tmp/cannon CHALLENGER=0 npx hardhat run scripts/respond.js --network localhost
done
