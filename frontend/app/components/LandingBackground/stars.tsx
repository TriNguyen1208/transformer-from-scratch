"use client"
import {motion} from "framer-motion"
import Image from "next/image"

function generateStars(count: number) {
    return Array.from({length: count}, (_, i) => ({
        id: i,
        top: Math.random() * 100,
        left: Math.random() * 100,
        duration: Math.random() * 3 + 2,
    }));
}

interface StarsProps {
    count: number;
}

export default function Stars({count}: StarsProps) {
    const stars = generateStars(count);

    return (
        <>
            {stars.map((star) => (
                <motion.div
                    key={star.id}
                    initial={{ opacity: 0.3 }}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{
                    duration: star.duration,
                    repeat: Infinity,
                    repeatType: "reverse",
                    }}
                    style={{ top: `${star.top}%`, left: `${star.left}%` }}
                    className="absolute z-10"
                >
                    
                    {/* <Image src="/assets/stars/star1.png" alt="star" width={16} height={16} /> */}
                    {/* <Image src="/assets/stars/star1.png" alt="star" width={16} height={16} /> */}
                </motion.div>
            ))}
        </>
    );
}